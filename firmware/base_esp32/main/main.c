// Beni base controller (§7.5): 100 Hz PID + odometry + safety on core 1, sensors/telemetry/protocol on core 0.
// Safety lives here and never depends on the Jetson: 300 ms command watchdog -> ramp to 0; bumper/cliff -> immediate
// stop with reverse-only allowed; front ultrasonic -> forward speed limit; e-stop line to Jetson pin 13.
#include <math.h>
#include <string.h>

#include "base_proto.h"
#include "config.h"
#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_system.h"
#include "esp_task_wdt.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "motor.h"
#include "sensors.h"

#define UART_PORT UART_NUM_2
#define TWO_PI 6.2831853f
#define M_PER_TICK (TWO_PI * WHEEL_RADIUS_M / TICKS_PER_REV)

typedef struct { uint8_t kind, arg; } event_t;

static portMUX_TYPE s_mux = portMUX_INITIALIZER_UNLOCKED;
static QueueHandle_t s_events;
// Host command + config (written by rx task, read by control task under s_mux).
static float s_cmd_v, s_cmd_w, s_v_max = DEFAULT_V_MAX, s_w_max = DEFAULT_W_MAX;
static float s_kp = DEFAULT_KP, s_ki = DEFAULT_KI, s_kd = DEFAULT_KD;
static int64_t s_last_cmd_us;
static bool s_estop_latched;
// Control -> telemetry snapshot (under s_mux).
static BaseStateMsg s_state;

static inline int64_t now_us(void) { return esp_timer_get_time(); }
static inline float clampf(float x, float lo, float hi) { return x < lo ? lo : x > hi ? hi : x; }
static inline int16_t sat16(float x) { return (int16_t)clampf(x, -32768.0f, 32767.0f); }
static void emit(uint8_t kind, uint8_t arg) {
  event_t e = {kind, arg};
  xQueueSend(s_events, &e, 0);
}
static void send(uint8_t type, const void* payload, uint8_t len) {
  static uint8_t seq;
  uint8_t buf[BASE_MAX_FRAME];
  size_t n = base_frame(type, seq++, payload, len, buf);
  uart_write_bytes(UART_PORT, buf, n);
}

// ---------------------------------------------------------------- core 1: control
static float ramp(float cur, float target, float up, float down, float dt) {
  // `up` when speeding up (|v| grows), `down` when slowing, so stops are always quicker than starts.
  float step = (fabsf(target) > fabsf(cur) && target * cur >= 0 ? up : down) * dt;
  return cur + clampf(target - cur, -step, step);
}

typedef struct { float integ, prev_e; } pid_t_;

static float pid_step(pid_t_* p, float target, float meas, float kp, float ki, float kd, float dt) {
  float t = target / MAX_WHEEL_MPS, e = t - meas / MAX_WHEEL_MPS;
  if (fabsf(t) < 1e-3f && fabsf(meas) < 0.02f) {  // parked: brake, no integrator creep
    p->integ = p->prev_e = 0.0f;
    return 0.0f;
  }
  float d = (e - p->prev_e) / dt;
  p->prev_e = e;
  float u = t + kp * e + ki * p->integ + kd * d;  // feed-forward + PID
  if (fabsf(u) < 1.0f || u * e < 0) p->integ = clampf(p->integ + e * dt, -0.5f, 0.5f);  // conditional anti-windup
  return clampf(u, -1.0f, 1.0f);
}

static void control_task(void* arg) {
  const float dt = 1.0f / CTRL_HZ;
  const float half = TRACK_M * 0.5f;
  int32_t enc_prev[2] = {encoder_read(0), encoder_read(1)};
  float meas[2] = {0, 0}, v_r = 0, w_r = 0, x = 0, y = 0, yaw = 0;
  pid_t_ pid[2] = {{0, 0}, {0, 0}};
  uint8_t prev_bump = 0, prev_cliff = 0;
  bool prev_wd = true, prev_estop = false;
  imu_sample_t imu = {0};
  esp_task_wdt_add(NULL);
  TickType_t wake = xTaskGetTickCount();
  for (;;) {
    vTaskDelayUntil(&wake, pdMS_TO_TICKS(1000 / CTRL_HZ));
    esp_task_wdt_reset();
    int64_t t = now_us();

    // Odometry: encoder deltas, midpoint integration.
    int32_t enc[2] = {encoder_read(0), encoder_read(1)};
    float d[2];
    for (int i = 0; i < 2; ++i) {
      d[i] = (enc[i] - enc_prev[i]) * M_PER_TICK;
      enc_prev[i] = enc[i];
      meas[i] = 0.6f * meas[i] + 0.4f * (d[i] / dt);  // 1-tick quantisation at low speed needs some smoothing
    }
    float ds = 0.5f * (d[0] + d[1]), dth = (d[1] - d[0]) / TRACK_M;
    x += ds * cosf(yaw + 0.5f * dth);
    y += ds * sinf(yaw + 0.5f * dth);
    yaw = remainderf(yaw + dth, TWO_PI);
    imu_read(&imu);

    // Snapshot host command.
    float cmd_v, cmd_w, v_max, w_max, kp, ki, kd;
    bool estop;
    int64_t last_cmd;
    portENTER_CRITICAL(&s_mux);
    cmd_v = s_cmd_v; cmd_w = s_cmd_w; v_max = s_v_max; w_max = s_w_max; kp = s_kp; ki = s_ki; kd = s_kd;
    estop = s_estop_latched; last_cmd = s_last_cmd_us;
    portEXIT_CRITICAL(&s_mux);

    // Safety.
    bool wd = (t - last_cmd) > WATCHDOG_MS * 1000LL;
    uint8_t bump = bumpers(), cliff = g_cliff;
    bool blocked = bump || cliff;  // reverse only
    if (wd) cmd_v = cmd_w = 0.0f;
    cmd_v = clampf(cmd_v, -v_max, v_max);
    cmd_w = clampf(cmd_w, -w_max, w_max);
    uint16_t us = 0xFFFF;
    for (int i = 0; i < 2; ++i)
      if (g_us_cm[i] && g_us_cm[i] < us) us = g_us_cm[i];
    if (cmd_v > 0 && us < US_SLOW_CM)
      cmd_v *= clampf((float)(us - US_STOP_CM) / (US_SLOW_CM - US_STOP_CM), 0.0f, 1.0f);

    v_r = ramp(v_r, cmd_v, ACCEL_MPS2, DECEL_MPS2, dt);
    w_r = ramp(w_r, cmd_w, ALPHA_RADPS2, 2.0f * ALPHA_RADPS2, dt);
    float tgt[2] = {v_r - w_r * half, v_r + w_r * half};
    if (blocked)
      for (int i = 0; i < 2; ++i)
        if (tgt[i] > 0) tgt[i] = 0;  // immediate: no ramp for forward motion into a hazard
    if (blocked && v_r > 0) v_r = 0;
    if (estop) {
      v_r = w_r = tgt[0] = tgt[1] = 0;
    }
    for (int i = 0; i < 2; ++i) {
      float u = pid_step(&pid[i], tgt[i], meas[i], kp, ki, kd, dt);
      if (blocked && u > 0) u = 0;  // PID overshoot must not push forward either
      motor_set(i, u);
    }

    uint8_t flags = (estop ? ST_ESTOP : 0) | ((bump & 1) ? ST_BUMP_L : 0) | ((bump & 2) ? ST_BUMP_R : 0) |
                    (cliff ? ST_CLIFF : 0) | (wd ? ST_WD_TIMEOUT : 0);
    gpio_set_level(PIN_ESTOP_OUT, estop || blocked);
    if (bump != prev_bump) emit(EV_BUMP, bump);
    if (cliff != prev_cliff) emit(EV_CLIFF, cliff);
    if (wd != prev_wd) emit(EV_WATCHDOG, wd);
    if (estop != prev_estop) emit(EV_ESTOP, estop);
    prev_bump = bump; prev_cliff = cliff; prev_wd = wd; prev_estop = estop;

    float v_meas = 0.5f * (meas[0] + meas[1]), w_meas = (meas[1] - meas[0]) / TRACK_M;
    portENTER_CRITICAL(&s_mux);
    s_state.t_ms = (uint32_t)(t / 1000);
    s_state.enc_l = enc[0]; s_state.enc_r = enc[1];
    s_state.v_mm_s = sat16(v_meas * 1000.0f); s_state.w_mrad_s = sat16(w_meas * 1000.0f);
    s_state.x_m = x; s_state.y_m = y; s_state.yaw_rad = yaw;
    s_state.gyro_z_mdps = imu.gyro_z_mdps;
    s_state.acc_x_mg = imu.acc_x_mg; s_state.acc_y_mg = imu.acc_y_mg; s_state.acc_z_mg = imu.acc_z_mg;
    s_state.flags = flags;
    portEXIT_CRITICAL(&s_mux);
  }
}

// ---------------------------------------------------------------- core 0: host link
static void handle(uint8_t type, uint8_t seq, const uint8_t* p, uint8_t len) {
  uint8_t ok = 1;
  switch (type) {
    case MSG_CMD_VEL: {
      if (len != sizeof(CmdVelMsg)) return;
      CmdVelMsg m;
      memcpy(&m, p, sizeof m);
      portENTER_CRITICAL(&s_mux);
      s_cmd_v = m.v_mm_s * 1e-3f;
      s_cmd_w = m.w_mrad_s * 1e-3f;
      s_last_cmd_us = now_us();
      portEXIT_CRITICAL(&s_mux);
      return;  // 20-50 Hz keep-alive: no ack
    }
    case MSG_CONFIG: {
      if (len != sizeof(ConfigMsg)) { ok = 0; break; }
      ConfigMsg c;
      memcpy(&c, p, sizeof c);
      if (!(c.kp >= 0 && c.ki >= 0 && c.kd >= 0 && c.v_max > 0 && c.v_max <= 1.0f && c.w_max > 0 && c.w_max <= 6.0f)) {
        ok = 0;
        break;
      }
      portENTER_CRITICAL(&s_mux);
      s_kp = c.kp; s_ki = c.ki; s_kd = c.kd; s_v_max = c.v_max; s_w_max = c.w_max;
      portEXIT_CRITICAL(&s_mux);
      break;
    }
    case MSG_ESTOP:
      portENTER_CRITICAL(&s_mux);
      s_estop_latched = len ? p[0] != 0 : true;
      if (s_estop_latched) s_cmd_v = s_cmd_w = 0;
      portEXIT_CRITICAL(&s_mux);
      break;
    default:
      ok = 0;  // MSG_SERVO / MSG_EXPRESSION / MSG_LED are Jetson-side in Revision 2
  }
  uint8_t ack[3] = {type, seq, ok};
  send(MSG_ACK, ack, 3);
}

static void rx_task(void* arg) {
  static uint8_t buf[BASE_MAX_FRAME], raw[BASE_MAX_FRAME], chunk[128];
  size_t n = 0;
  for (;;) {
    int got = uart_read_bytes(UART_PORT, chunk, sizeof chunk, pdMS_TO_TICKS(10));
    for (int i = 0; i < got; ++i) {
      if (chunk[i] != 0) {
        if (n < sizeof buf) buf[n++] = chunk[i];
        else n = sizeof buf + 1;  // overlong: drop until the next delimiter
        continue;
      }
      size_t m = (n && n <= sizeof buf) ? base_cobs_decode(buf, n, raw) : 0;
      n = 0;
      if (m < 5 || raw[2] != m - 5) continue;
      uint16_t crc = (uint16_t)(raw[m - 2] | (raw[m - 1] << 8));
      if (base_crc16(raw, m - 2) != crc) continue;
      handle(raw[0], raw[1], raw + 3, raw[2]);
    }
  }
}

static void io_task(void* arg) {
  esp_task_wdt_add(NULL);
  TickType_t wake = xTaskGetTickCount();
  bool low = false;
  for (uint32_t tick = 0;; ++tick) {
    vTaskDelayUntil(&wake, pdMS_TO_TICKS(1000 / STATE_HZ));
    esp_task_wdt_reset();
    ranging_step();
    if (tick % (STATE_HZ / 10) == 0) {
      battery_step();
      int pct = (int)clampf(100.0f * (g_batt_mv - BATT_EMPTY_MV) / (BATT_FULL_MV - BATT_EMPTY_MV), 0, 100);
      if (!low && g_batt_mv && pct < BATT_LOW_PCT) {
        emit(EV_LOW_BATT, (uint8_t)pct);
        low = true;
      } else if (low && pct > BATT_LOW_PCT + 5) {
        low = false;  // hysteresis
      }
    }
    BaseStateMsg st;
    portENTER_CRITICAL(&s_mux);
    st = s_state;
    portEXIT_CRITICAL(&s_mux);
    st.batt_mv = g_batt_mv;
    st.batt_ma = g_batt_ma;
    if (g_batt_mv > BATT_CHARGING_MV) st.flags |= ST_CHARGING;
    st.range_cm[0] = (uint8_t)(g_tof_mm[0] / 10 > 255 ? 255 : g_tof_mm[0] / 10);
    st.range_cm[1] = (uint8_t)(g_tof_mm[1] / 10 > 255 ? 255 : g_tof_mm[1] / 10);
    st.range_cm[2] = (uint8_t)(g_us_cm[0] > 255 ? 255 : g_us_cm[0]);
    st.range_cm[3] = (uint8_t)(g_us_cm[1] > 255 ? 255 : g_us_cm[1]);
    send(MSG_STATE, &st, sizeof st);
    event_t e;
    while (xQueueReceive(s_events, &e, 0)) send(MSG_EVENT, &e, 2);
  }
}

void app_main(void) {
  _Static_assert(sizeof(BaseStateMsg) == 45, "BaseStateMsg must stay 45 bytes (shared/beni_common/proto.py)");
  gpio_config_t io = {.pin_bit_mask = 1ULL << PIN_ESTOP_OUT, .mode = GPIO_MODE_OUTPUT};
  gpio_config(&io);
  gpio_set_level(PIN_ESTOP_OUT, 1);
  uart_config_t uc = {.baud_rate = UART_BAUD, .data_bits = UART_DATA_8_BITS, .parity = UART_PARITY_DISABLE,
                      .stop_bits = UART_STOP_BITS_1, .flow_ctrl = UART_HW_FLOWCTRL_DISABLE, .source_clk = UART_SCLK_DEFAULT};
  ESP_ERROR_CHECK(uart_driver_install(UART_PORT, 2048, 2048, 0, NULL, 0));
  ESP_ERROR_CHECK(uart_param_config(UART_PORT, &uc));
  ESP_ERROR_CHECK(uart_set_pin(UART_PORT, PIN_UART_TX, PIN_UART_RX, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
  s_events = xQueueCreate(16, sizeof(event_t));
  motor_init();
  sensors_init();
  s_last_cmd_us = now_us() - WATCHDOG_MS * 1000LL;  // start in watchdog-stopped state
  emit(EV_BOOT, (uint8_t)esp_reset_reason());
  if (g_sensor_fault) emit(EV_SENSOR_FAULT, g_sensor_fault);
  xTaskCreatePinnedToCore(control_task, "ctrl", 4096, NULL, configMAX_PRIORITIES - 2, NULL, 1);
  xTaskCreatePinnedToCore(rx_task, "rx", 3072, NULL, 12, NULL, 0);
  xTaskCreatePinnedToCore(io_task, "io", 4096, NULL, 10, NULL, 0);
}
