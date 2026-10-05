// Sensors on the base ESP32 (§7.5). All I2C devices share one bus at 400 kHz; the i2c_master driver serialises access
// between the IMU (core 1) and the ToF/INA219 polling (core 0).
#include "sensors.h"

#include "config.h"
#include "driver/gpio.h"
#include "driver/i2c_master.h"
#include "esp_adc/adc_cali.h"
#include "esp_adc/adc_cali_scheme.h"
#include "esp_adc/adc_oneshot.h"
#include "esp_rom_sys.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "vl53l0x.h"

volatile uint8_t g_cliff = 3, g_sensor_fault;  // cliff until the first valid ToF reading
volatile uint16_t g_us_cm[2], g_tof_mm[2], g_batt_mv;
volatile int16_t g_batt_ma;

static i2c_master_bus_handle_t s_bus;
static i2c_master_dev_handle_t s_imu, s_ina;
static vl53l0x_t s_tof[2];
static float s_gyro_bias;
static adc_oneshot_unit_handle_t s_adc;
static adc_cali_handle_t s_cali;
static const int s_trig[2] = {PIN_US_TRIG_L, PIN_US_TRIG_R}, s_echo[2] = {PIN_US_ECHO_L, PIN_US_ECHO_R};
static volatile int64_t s_echo_t0[2];
static volatile uint16_t s_echo_cm[2];
static int s_us_turn;

#define GYRO_LSB_PER_DPS 65.5f  // +-500 dps
#define ACC_LSB_PER_G 8192.0f   // +-4 g
#define DEG2RAD 0.017453292f

static esp_err_t reg_wr(i2c_master_dev_handle_t d, uint8_t reg, uint8_t v) {
  uint8_t b[2] = {reg, v};
  return i2c_master_transmit(d, b, 2, 5);
}

static bool imu_raw(int16_t v[7]) {
  uint8_t reg = 0x3B, b[14];
  if (i2c_master_transmit_receive(s_imu, &reg, 1, b, 14, 3) != ESP_OK) return false;
  for (int i = 0; i < 7; ++i) v[i] = (int16_t)((b[2 * i] << 8) | b[2 * i + 1]);  // ax ay az temp gx gy gz
  return true;
}

static void imu_init(void) {
  i2c_device_config_t dc = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = 0x68, .scl_speed_hz = 400000};
  i2c_master_bus_add_device(s_bus, &dc, &s_imu);
  bool ok = reg_wr(s_imu, 0x6B, 0x80) == ESP_OK;  // reset
  vTaskDelay(pdMS_TO_TICKS(100));
  ok = ok && reg_wr(s_imu, 0x6B, 0x01) == ESP_OK   // PLL on gyro X
       && reg_wr(s_imu, 0x1A, 0x03) == ESP_OK      // DLPF 44 Hz
       && reg_wr(s_imu, 0x19, 0x04) == ESP_OK      // 200 Hz sample rate
       && reg_wr(s_imu, 0x1B, 0x08) == ESP_OK      // +-500 dps
       && reg_wr(s_imu, 0x1C, 0x08) == ESP_OK;     // +-4 g
  if (!ok) {
    g_sensor_fault |= 1;
    return;
  }
  vTaskDelay(pdMS_TO_TICKS(50));
  int16_t v[7];
  int32_t sum = 0, n = 0;
  for (int i = 0; i < 200; ++i, vTaskDelay(pdMS_TO_TICKS(5)))  // the robot is stationary at boot
    if (imu_raw(v)) sum += v[6], ++n;
  s_gyro_bias = n ? (float)sum / n : 0.0f;
}

bool imu_read(imu_sample_t* o) {
  int16_t v[7];
  if ((g_sensor_fault & 1) || !imu_raw(v)) return false;
  float gz_dps = (v[6] - s_gyro_bias) / GYRO_LSB_PER_DPS;
  o->gyro_z_mdps = (int16_t)(gz_dps * 1000.0f > 32767 ? 32767 : gz_dps * 1000.0f < -32768 ? -32768 : gz_dps * 1000.0f);
  o->gyro_z_rad = gz_dps * DEG2RAD;
  o->acc_x_mg = (int16_t)(v[0] * 1000.0f / ACC_LSB_PER_G);
  o->acc_y_mg = (int16_t)(v[1] * 1000.0f / ACC_LSB_PER_G);
  o->acc_z_mg = (int16_t)(v[2] * 1000.0f / ACC_LSB_PER_G);
  return true;
}

static void tof_init(void) {
  gpio_config_t io = {.pin_bit_mask = (1ULL << PIN_XSHUT_L) | (1ULL << PIN_XSHUT_R), .mode = GPIO_MODE_OUTPUT};
  gpio_config(&io);
  gpio_set_level(PIN_XSHUT_L, 0);
  gpio_set_level(PIN_XSHUT_R, 0);
  vTaskDelay(pdMS_TO_TICKS(10));
  const int xshut[2] = {PIN_XSHUT_L, PIN_XSHUT_R};
  const uint8_t addr[2] = {TOF_ADDR_L, TOF_ADDR_R};
  for (int i = 0; i < 2; ++i) {  // wake one at a time; each moves off 0x29 before the next boots
    gpio_set_level(xshut[i], 1);
    vTaskDelay(pdMS_TO_TICKS(5));
    if (!vl53l0x_init(&s_tof[i], s_bus, addr[i])) g_sensor_fault |= 2 << i;
  }
#if ALLOW_MISSING_TOF
  g_cliff = 0;
#endif
}

static void IRAM_ATTR echo_isr(void* arg) {
  int i = (int)(intptr_t)arg;
  int64_t now = esp_timer_get_time();
  if (gpio_get_level(s_echo[i])) {
    s_echo_t0[i] = now;
  } else if (s_echo_t0[i]) {
    int64_t us = now - s_echo_t0[i];
    s_echo_cm[i] = us < 20000 ? (uint16_t)(us / 58) : 0;
    s_echo_t0[i] = 0;
  }
}

static void ultrasonic_init(void) {
  gpio_config_t out = {.pin_bit_mask = (1ULL << PIN_US_TRIG_L) | (1ULL << PIN_US_TRIG_R), .mode = GPIO_MODE_OUTPUT};
  gpio_config(&out);
  gpio_config_t in = {.pin_bit_mask = (1ULL << PIN_US_ECHO_L) | (1ULL << PIN_US_ECHO_R), .mode = GPIO_MODE_INPUT,
                      .intr_type = GPIO_INTR_ANYEDGE};
  gpio_config(&in);
  gpio_install_isr_service(ESP_INTR_FLAG_IRAM);
  for (int i = 0; i < 2; ++i) gpio_isr_handler_add(s_echo[i], echo_isr, (void*)(intptr_t)i);
}

static void battery_init(void) {
  adc_oneshot_unit_init_cfg_t u = {.unit_id = ADC_UNIT_1};
  adc_oneshot_new_unit(&u, &s_adc);
  adc_oneshot_chan_cfg_t c = {.atten = ADC_ATTEN_DB_12, .bitwidth = ADC_BITWIDTH_12};
  adc_oneshot_config_channel(s_adc, ADC_CHANNEL_4, &c);  // GPIO32
  adc_cali_line_fitting_config_t cc = {.unit_id = ADC_UNIT_1, .atten = ADC_ATTEN_DB_12, .bitwidth = ADC_BITWIDTH_12};
  if (adc_cali_create_scheme_line_fitting(&cc, &s_cali) != ESP_OK) s_cali = NULL;
  if (i2c_master_probe(s_bus, 0x40, 5) == ESP_OK) {  // optional INA219 on the battery lead
    i2c_device_config_t dc = {.dev_addr_length = I2C_ADDR_BIT_LEN_7, .device_address = 0x40, .scl_speed_hz = 400000};
    i2c_master_bus_add_device(s_bus, &dc, &s_ina);
  }
}

void sensors_init(void) {
  i2c_master_bus_config_t bc = {.i2c_port = I2C_NUM_0, .sda_io_num = PIN_SDA, .scl_io_num = PIN_SCL,
                                .clk_source = I2C_CLK_SRC_DEFAULT, .glitch_ignore_cnt = 7,
                                .flags.enable_internal_pullup = true};
  ESP_ERROR_CHECK(i2c_new_master_bus(&bc, &s_bus));
  gpio_config_t bump = {.pin_bit_mask = (1ULL << PIN_BUMP_L) | (1ULL << PIN_BUMP_R), .mode = GPIO_MODE_INPUT,
                        .pull_up_en = GPIO_PULLUP_ENABLE};
  gpio_config(&bump);
  tof_init();
  imu_init();
  ultrasonic_init();
  battery_init();
}

uint8_t bumpers(void) { return (uint8_t)((!gpio_get_level(PIN_BUMP_L)) | ((!gpio_get_level(PIN_BUMP_R)) << 1)); }

void ranging_step(void) {
  uint8_t cliff = g_cliff;
  for (int i = 0; i < 2; ++i) {
    if (!s_tof[i].ok) {
      if (!ALLOW_MISSING_TOF) cliff |= 1 << i;
      continue;
    }
    int mm = vl53l0x_poll(&s_tof[i]);
    if (mm < 0) continue;
    g_tof_mm[i] = (uint16_t)mm;
    if (mm > CLIFF_MM) cliff |= 1 << i;  // includes 8190/8191 "nothing in range"
    else cliff &= ~(1 << i);
  }
  g_cliff = cliff;
  // One HC-SR04 per tick (alternating, 20 ms echo window = 3.4 m) so they never hear each other: 25 Hz each.
  int i = s_us_turn;
  s_us_turn ^= 1;
  if (s_echo_t0[i]) s_echo_cm[i] = 0;  // previous ping never came back
  g_us_cm[i] = s_echo_cm[i];
  s_echo_t0[i] = 0;
  gpio_set_level(s_trig[i], 1);
  esp_rom_delay_us(10);
  gpio_set_level(s_trig[i], 0);
}

void battery_step(void) {
  static float mv_f;
  float mv = 0;
  if (s_ina) {
    uint8_t reg = 0x02, b[2];
    if (i2c_master_transmit_receive(s_ina, &reg, 1, b, 2, 3) == ESP_OK) mv = (float)(((b[0] << 8) | b[1]) >> 3) * 4.0f;
    reg = 0x01;
    if (i2c_master_transmit_receive(s_ina, &reg, 1, b, 2, 3) == ESP_OK)
      g_batt_ma = (int16_t)((int16_t)((b[0] << 8) | b[1]) * 10 / INA219_SHUNT_MOHM);  // 10 uV LSB / mOhm = mA
  }
  if (mv <= 0) {
    int raw = 0, pin_mv = 0;
    if (adc_oneshot_read(s_adc, ADC_CHANNEL_4, &raw) != ESP_OK) return;
    if (!s_cali || adc_cali_raw_to_voltage(s_cali, raw, &pin_mv) != ESP_OK) pin_mv = raw * 3100 / 4095;
    mv = pin_mv * BATT_DIV * BATT_CAL;
  }
  mv_f = mv_f ? 0.9f * mv_f + 0.1f * mv : mv;
  g_batt_mv = (uint16_t)mv_f;
}
