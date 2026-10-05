// TB6612 + encoders (§7.5). PCNT does x4 quadrature decoding in hardware, so the CPU never sees encoder edges.
#include "motor.h"

#include <math.h>

#include "config.h"
#include "driver/gpio.h"
#include "driver/ledc.h"
#include "driver/pulse_cnt.h"

static pcnt_unit_handle_t s_unit[2];
static const int s_pwm[2] = {PIN_PWMA, PIN_PWMB}, s_in1[2] = {PIN_AIN1, PIN_BIN1}, s_in2[2] = {PIN_AIN2, PIN_BIN2};
static const int s_enc_a[2] = {PIN_ENC_LA, PIN_ENC_RA}, s_enc_b[2] = {PIN_ENC_LB, PIN_ENC_RB};
static const float s_sign[2] = {1.0f, -1.0f};  // right motor is mirrored on the chassis
#define DUTY_MAX ((1u << PWM_BITS) - 1)

static void encoder_init(int side) {
  pcnt_unit_config_t ucfg = {.low_limit = -30000, .high_limit = 30000, .flags.accum_count = 1};
  ESP_ERROR_CHECK(pcnt_new_unit(&ucfg, &s_unit[side]));
  pcnt_glitch_filter_config_t filt = {.max_glitch_ns = 1000};  // edges are >= 50 us apart at full speed
  ESP_ERROR_CHECK(pcnt_unit_set_glitch_filter(s_unit[side], &filt));
  pcnt_chan_config_t ca = {.edge_gpio_num = s_enc_a[side], .level_gpio_num = s_enc_b[side]};
  pcnt_chan_config_t cb = {.edge_gpio_num = s_enc_b[side], .level_gpio_num = s_enc_a[side]};
  pcnt_channel_handle_t cha, chb;
  ESP_ERROR_CHECK(pcnt_new_channel(s_unit[side], &ca, &cha));
  ESP_ERROR_CHECK(pcnt_new_channel(s_unit[side], &cb, &chb));
  pcnt_channel_set_edge_action(cha, PCNT_CHANNEL_EDGE_ACTION_DECREASE, PCNT_CHANNEL_EDGE_ACTION_INCREASE);
  pcnt_channel_set_level_action(cha, PCNT_CHANNEL_LEVEL_ACTION_KEEP, PCNT_CHANNEL_LEVEL_ACTION_INVERSE);
  pcnt_channel_set_edge_action(chb, PCNT_CHANNEL_EDGE_ACTION_INCREASE, PCNT_CHANNEL_EDGE_ACTION_DECREASE);
  pcnt_channel_set_level_action(chb, PCNT_CHANNEL_LEVEL_ACTION_KEEP, PCNT_CHANNEL_LEVEL_ACTION_INVERSE);
  pcnt_unit_add_watch_point(s_unit[side], ucfg.low_limit);   // accum_count needs the limits as watch points
  pcnt_unit_add_watch_point(s_unit[side], ucfg.high_limit);
  ESP_ERROR_CHECK(pcnt_unit_enable(s_unit[side]));
  ESP_ERROR_CHECK(pcnt_unit_clear_count(s_unit[side]));
  ESP_ERROR_CHECK(pcnt_unit_start(s_unit[side]));
}

void motor_init(void) {
  ledc_timer_config_t t = {.speed_mode = LEDC_HIGH_SPEED_MODE, .duty_resolution = PWM_BITS, .timer_num = LEDC_TIMER_0,
                           .freq_hz = PWM_HZ, .clk_cfg = LEDC_AUTO_CLK};
  ESP_ERROR_CHECK(ledc_timer_config(&t));
  uint64_t mask = 1ULL << PIN_STBY;
  for (int i = 0; i < 2; ++i) {
    ledc_channel_config_t c = {.gpio_num = s_pwm[i], .speed_mode = LEDC_HIGH_SPEED_MODE, .channel = (ledc_channel_t)i,
                               .timer_sel = LEDC_TIMER_0, .duty = 0, .hpoint = 0};
    ESP_ERROR_CHECK(ledc_channel_config(&c));
    mask |= (1ULL << s_in1[i]) | (1ULL << s_in2[i]);
  }
  gpio_config_t io = {.pin_bit_mask = mask, .mode = GPIO_MODE_OUTPUT};
  ESP_ERROR_CHECK(gpio_config(&io));
  for (int i = 0; i < 2; ++i) {
    motor_set(i, 0.0f);
    encoder_init(i);
  }
  motor_enable(true);
}

void motor_enable(bool on) { gpio_set_level(PIN_STBY, on); }

void motor_set(int side, float u) {
  u *= s_sign[side];
  if (u > 1.0f) u = 1.0f;
  if (u < -1.0f) u = -1.0f;
  uint32_t duty = (uint32_t)(fabsf(u) * DUTY_MAX + 0.5f);
  if (duty < 8) {  // short brake: both inputs high, PWM high
    gpio_set_level(s_in1[side], 1);
    gpio_set_level(s_in2[side], 1);
    duty = 0;
  } else {
    gpio_set_level(s_in1[side], u > 0);
    gpio_set_level(s_in2[side], u < 0);
  }
  ledc_set_duty(LEDC_HIGH_SPEED_MODE, (ledc_channel_t)side, duty);
  ledc_update_duty(LEDC_HIGH_SPEED_MODE, (ledc_channel_t)side);
}

int32_t encoder_read(int side) {
  int v = 0;
  pcnt_unit_get_count(s_unit[side], &v);
  return s_sign[side] < 0 ? -v : v;  // integer: a float product drops odd ticks past 2^24 (~3.6 km)
}
