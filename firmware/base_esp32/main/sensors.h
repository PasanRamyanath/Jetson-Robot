// IMU, cliff ToF, ultrasonic, bumpers, battery (§7.5).
#pragma once
#include <stdbool.h>
#include <stdint.h>

typedef struct {
  int16_t gyro_z_mdps, acc_x_mg, acc_y_mg, acc_z_mg;
  float gyro_z_rad;  // bias-corrected
} imu_sample_t;

void sensors_init(void);                // I2C bus, IMU (+gyro bias), ToF x2, INA219, ultrasonic ISRs, ADC, bumpers
bool imu_read(imu_sample_t* out);       // core 1, 100 Hz
void ranging_step(void);                // core 0, 50 Hz: ToF polls + one ultrasonic ping (alternating)
void battery_step(void);                // core 0, 10 Hz
uint8_t bumpers(void);                  // bit0 left, bit1 right (instant GPIO read)

// Latest values written by ranging_step / battery_step (single-writer, word-sized: read without locks).
extern volatile uint8_t g_cliff;        // bit0 left, bit1 right
extern volatile uint16_t g_us_cm[2];    // 0 = no echo
extern volatile uint16_t g_tof_mm[2];
extern volatile uint16_t g_batt_mv;
extern volatile int16_t g_batt_ma;
extern volatile uint8_t g_sensor_fault; // bit0 imu, bit1 tof_l, bit2 tof_r
