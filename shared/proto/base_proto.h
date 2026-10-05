// ESP32 <-> Jetson base protocol (§7.5). Shared by firmware/base_esp32 and ros2 beni_base_driver.
// Frame: COBS( [type u8][seq u8][len u8][payload ...][crc16 u16 LE] ) 0x00
// CRC16-CCITT (poly 0x1021, init 0xFFFF) over type..payload. Mirror: shared/beni_common/proto.py
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  MSG_CMD_VEL = 0x01,     // host->mcu: int16 v_mm_s, int16 w_mrad_s
  MSG_SERVO = 0x02,       // unused in Rev 2 (head is on Jetson I2C)
  MSG_EXPRESSION = 0x03,  // unused in Rev 2 (face is HDMI)
  MSG_LED = 0x04,         // host->mcu: uint8 mode, r, g, b
  MSG_CONFIG = 0x05,      // host->mcu: float32 kp, ki, kd, v_max, w_max
  MSG_ESTOP = 0x06,       // both ways
  MSG_STATE = 0x81,       // mcu->host 50 Hz: BaseStateMsg
  MSG_EVENT = 0x82,       // mcu->host: uint8 kind, uint8 arg
  MSG_ACK = 0x83,
  MSG_LOG = 0x84,
};

enum {
  ST_ESTOP = 1 << 0, ST_BUMP_L = 1 << 1, ST_BUMP_R = 1 << 2,
  ST_CLIFF = 1 << 3, ST_CHARGING = 1 << 4, ST_WD_TIMEOUT = 1 << 5,
};

// MSG_EVENT kinds (arg meaning in brackets). MSG_ACK payload: uint8 acked type, uint8 acked seq, uint8 ok.
enum {
  EV_BOOT = 1,       // [esp_reset_reason]
  EV_BUMP = 2,       // [bit0 left, bit1 right]
  EV_CLIFF = 3,      // [bit0 left, bit1 right]
  EV_ESTOP = 4,      // [1 engaged, 0 cleared]
  EV_LOW_BATT = 5,   // [percent]
  EV_WATCHDOG = 6,   // [1 timed out, 0 recovered]
  EV_SENSOR_FAULT = 7,  // [bit0 imu, bit1 tof_l, bit2 tof_r]
};

#pragma pack(push, 1)
typedef struct { int16_t v_mm_s, w_mrad_s; } CmdVelMsg;
typedef struct { float kp, ki, kd, v_max, w_max; } ConfigMsg;  // v_max m/s, w_max rad/s
typedef struct {
  uint32_t t_ms;
  int32_t enc_l, enc_r;          // cumulative ticks
  int16_t v_mm_s, w_mrad_s;      // measured
  float x_m, y_m, yaw_rad;       // integrated odometry
  int16_t gyro_z_mdps, acc_x_mg, acc_y_mg, acc_z_mg;
  uint16_t batt_mv;
  int16_t batt_ma;
  uint8_t flags;                 // ST_* bits
  uint8_t range_cm[4];           // ultrasonic / ToF
} BaseStateMsg;                  // 45 bytes
#pragma pack(pop)

#define BASE_MAX_PAYLOAD 255
#define BASE_MAX_FRAME (3 + BASE_MAX_PAYLOAD + 2 + 2 + 1)

static inline uint16_t base_crc16(const uint8_t* p, size_t n) {
  uint16_t crc = 0xFFFF;
  while (n--) {
    crc ^= (uint16_t)(*p++) << 8;
    for (int i = 0; i < 8; ++i) crc = (crc & 0x8000) ? (uint16_t)((crc << 1) ^ 0x1021) : (uint16_t)(crc << 1);
  }
  return crc;
}

// COBS-encode n bytes from src into dst (needs n + n/254 + 1 bytes). Returns encoded length (no 0x00).
static inline size_t base_cobs_encode(const uint8_t* src, size_t n, uint8_t* dst) {
  size_t out = 1, code_at = 0;
  uint8_t code = 1;
  for (size_t i = 0; i < n; ++i) {
    if (src[i] == 0) {
      dst[code_at] = code; code_at = out++; code = 1;
    } else {
      dst[out++] = src[i];
      if (++code == 0xFF) { dst[code_at] = code; code_at = out++; code = 1; }
    }
  }
  dst[code_at] = code;
  return out;
}

// Decode n bytes (without the trailing 0x00). Returns decoded length or 0 on error.
static inline size_t base_cobs_decode(const uint8_t* src, size_t n, uint8_t* dst) {
  size_t i = 0, out = 0;
  while (i < n) {
    uint8_t code = src[i];
    if (code == 0 || i + code > n) return 0;
    for (uint8_t k = 1; k < code; ++k) dst[out++] = src[i + k];
    i += code;
    if (code < 0xFF && i < n) dst[out++] = 0;
  }
  return out;
}

// Build a complete frame (incl. trailing 0x00) into out[BASE_MAX_FRAME]. Returns bytes to send.
static inline size_t base_frame(uint8_t type, uint8_t seq, const void* payload, uint8_t len, uint8_t* out) {
  uint8_t raw[3 + BASE_MAX_PAYLOAD + 2];
  raw[0] = type; raw[1] = seq; raw[2] = len;
  for (uint8_t i = 0; i < len; ++i) raw[3 + i] = ((const uint8_t*)payload)[i];
  uint16_t crc = base_crc16(raw, 3u + len);
  raw[3 + len] = (uint8_t)(crc & 0xFF); raw[4 + len] = (uint8_t)(crc >> 8);
  size_t n = base_cobs_encode(raw, 5u + len, out);
  out[n++] = 0;
  return n;
}

#ifdef __cplusplus
}
#endif
