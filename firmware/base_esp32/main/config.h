// Pins, geometry and limits for the base controller (§7.5, wiring §13.4). ESP32-WROOM-32 DevKit (38-pin).
#pragma once

// Jetson UART (UART2): 921600 8N1, header pins 8/10.
#define PIN_UART_RX 16
#define PIN_UART_TX 17
#define UART_BAUD 921600

// TB6612FNG
#define PIN_PWMA 25
#define PIN_AIN1 26
#define PIN_AIN2 27
#define PIN_PWMB 14
#define PIN_BIN1 12  // strap pin (MTDI): TB6612 inputs are high-Z at boot, so flash voltage stays 3.3 V
#define PIN_BIN2 13
#define PIN_STBY 33

// Quadrature encoders (input-only pins; the encoder boards have pull-ups)
#define PIN_ENC_LA 34
#define PIN_ENC_LB 35
#define PIN_ENC_RA 36
#define PIN_ENC_RB 39

// I2C: MPU6050 (0x68), VL53L0X x2 (re-addressed to 0x30/0x31 via XSHUT), optional INA219 (0x40)
#define PIN_SDA 21
#define PIN_SCL 22
#define PIN_XSHUT_L 4
#define PIN_XSHUT_R 5
#define TOF_ADDR_L 0x30
#define TOF_ADDR_R 0x31

// HC-SR04 front pair (echo through a 1k/2k divider)
#define PIN_US_TRIG_L 18
#define PIN_US_ECHO_L 19
#define PIN_US_TRIG_R 23
#define PIN_US_ECHO_R 15

// §13.4 lists GPIO0/13 for bumpers, but 13 is BIN2; the only free pins left are 0 and 3 (UART0 RX, unused at runtime).
#define PIN_BUMP_L 0
#define PIN_BUMP_R 3

#define PIN_BATT_ADC 32  // ADC1 channel 4, 100k/22k divider
#define PIN_ESTOP_OUT 2  // -> Jetson header pin 13; high while any safety stop is active (also the on-board LED)

// Geometry: JGA25-370 280 RPM, 65 mm wheels. Verify TICKS_PER_REV by turning a wheel 10 revs (§7.5).
#define TICKS_PER_REV 937.0f
#define WHEEL_RADIUS_M 0.0325f
#define TRACK_M 0.170f
#define MAX_WHEEL_MPS 0.95f  // 280 RPM * 2*pi*r / 60, used for feed-forward

// Control
#define CTRL_HZ 100
#define STATE_HZ 50
#define PWM_HZ 20000
#define PWM_BITS 11
#define WATCHDOG_MS 300
#define ACCEL_MPS2 0.8f       // current ramp: limits peak current and battery sag (§15.2 item 8)
#define DECEL_MPS2 1.6f
#define ALPHA_RADPS2 4.0f
#define DEFAULT_V_MAX 0.5f    // m/s; the host lowers it further (0.4 indoors)
#define DEFAULT_W_MAX 2.5f    // rad/s
#define DEFAULT_KP 0.9f       // on normalised wheel speed error (1.0 = MAX_WHEEL_MPS)
#define DEFAULT_KI 6.0f
#define DEFAULT_KD 0.0f

// Safety thresholds
#define CLIFF_MM 70           // floor further than this below a ToF sensor = cliff
#define US_STOP_CM 12         // forward motion blocked inside this
#define US_SLOW_CM 45         // forward speed scales down linearly from here
#define ALLOW_MISSING_TOF 0   // 1 only on the bench: a missing cliff sensor otherwise reads as a cliff

// Battery (3S Li-ion)
#define BATT_DIV ((100.0f + 22.0f) / 22.0f)
#define BATT_CAL 1.000f       // multimeter / reported
#define BATT_FULL_MV 12600
#define BATT_EMPTY_MV 9900
#define BATT_CHARGING_MV 12700  // only a charger pushes the pack above this
#define BATT_LOW_PCT 20
#define INA219_SHUNT_MOHM 100
