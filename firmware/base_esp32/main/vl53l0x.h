// Minimal VL53L0X driver (register-level port of the Pololu init sequence, continuous back-to-back mode).
#pragma once
#include <stdbool.h>
#include <stdint.h>

#include "driver/i2c_master.h"

typedef struct {
  i2c_master_dev_handle_t dev;
  uint8_t stop_variable;
  bool ok;
} vl53l0x_t;

// Brings up the sensor that currently answers at 0x29, moves it to new_addr, starts continuous ranging.
bool vl53l0x_init(vl53l0x_t* s, i2c_master_bus_handle_t bus, uint8_t new_addr);
// Non-blocking: returns the range in mm (8190+ = nothing in range) or -1 when no new sample is ready.
int vl53l0x_poll(vl53l0x_t* s);
