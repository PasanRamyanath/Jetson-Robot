// TB6612 PWM (LEDC 20 kHz) + PCNT quadrature encoders (§7.5).
#pragma once
#include <stdbool.h>
#include <stdint.h>

void motor_init(void);
void motor_enable(bool on);               // STBY pin
void motor_set(int side, float u);        // side 0 = left, 1 = right; u in [-1, 1]; 0 = active brake
int32_t encoder_read(int side);           // cumulative ticks (overflow-compensated)
