# 5. ESP32 base firmware (§7.5)

The ESP32 runs these loops:
- the 100 Hz wheel PID and odometry;
- the safety layer: 300 ms watchdog, bumpers, cliff ToF sensors and ultrasonic slow-down;
- 50 Hz state frames to the Jetson over UART at 921600 baud.

It uses **ESP-IDF 5.3.1** (not Arduino). The Jetson side is `beni_common.proto`; the C and Python frame codecs are
checked against each other in `tests/contract`.

## 5.1 Wiring (from `firmware/base_esp32/main/config.h`)

| Function | ESP32 GPIO | Goes to |
|---|---|---|
| UART2 RX / TX (921600 8N1) | 16 / 17 | Jetson header **pin 8** (TXD) / **pin 10** (RXD), i.e. `/dev/ttyTHS1`; common GND |
| TB6612 PWMA, AIN1, AIN2 | 25, 26, 27 | left motor |
| TB6612 PWMB, BIN1, BIN2 | 14, 12, 13 | right motor (GPIO12 is a strap pin; fine with the TB6612) |
| TB6612 STBY | 33 | |
| Encoders L A/B, R A/B | 34, 35, 36, 39 | input-only pins; add external pull-ups if the encoders are open-collector |
| I2C SDA / SCL | 21 / 22 | MPU6050 (0x68), 2× VL53L0X (XSHUT on 4 / 5, re-addressed to 0x30 / 0x31), optional INA219 (0x40) |
| Ultrasonic L trig/echo, R trig/echo | 18/19, 23/15 | 5 V HC-SR04 echo needs a divider down to 3.3 V |
| Bumpers L / R | 0 / 3 | switches to GND |
| Battery ADC | 32 | 100 k / 22 k divider from the 3S pack |
| E-stop out | 2 | Jetson header **pin 13** (sysfs gpio14, `BENI_GPIO_ESTOP`); high while any safety stop is active |

Both boards run 3.3 V logic, so no level shifter is needed on the UART or the e-stop line. `beni-agent`'s
`ExecStartPre` exports pin 13 as an edge-triggered input.

Before building, set the robot's geometry in `config.h`:
- `TICKS_PER_REV`, `WHEEL_RADIUS_M`, `TRACK_M`, `MAX_WHEEL_MPS`;
- `BATT_CAL` (multimeter reading ÷ reported value).

## 5.2 Build and flash

**PlatformIO (easiest):** the pinned `espressif32@6.9.0` platform brings ESP-IDF 5.3.1.

```bash
pc$ pip install -U platformio
pc$ cd ~/beni/firmware/base_esp32
pc$ pio run                                   # first build downloads the toolchain (~1 GB, in ~/.platformio)
pc$ pio run -t upload --upload-port /dev/ttyUSB0      # Windows: COM5
pc$ pio device monitor                        # 115200, boot log on UART0 (USB); warnings only
```

**ESP-IDF directly:**

```bash
pc$ . ~/esp/esp-idf-v5.3.1/export.sh
pc$ cd ~/beni/firmware/base_esp32 && idf.py set-target esp32 && idf.py build
pc$ idf.py -p /dev/ttyUSB0 flash monitor
```

The toolchain needs ~1–2 GB. On a PC with little space on the system disk, point it elsewhere first:
`PLATFORMIO_CORE_DIR=D:\pio` for PlatformIO, or `IDF_TOOLS_PATH` for ESP-IDF.

You can also flash from the Nano: it has `/dev/ttyUSB0` when the ESP32 is on USB. `pip3 install --user platformio`
works there too, though slowly.

## 5.3 Bench test from the Nano (wheels off the ground)

`jetson-setup` already disabled `nvgetty` on the UART and put you in `dialout`, and it installs pyserial for the
host Python 3.6. Stop the ROS bridge so it doesn't hold the port:

```bash
nano$ sudo systemctl stop beni-ros
nano$ cd ~/beni
nano$ python3 jetson/tools/base_console.py monitor        # ~5 lines/s: t, v, w, x, y, yaw, batt, flags, range, err
nano$ python3 jetson/tools/base_console.py drive 200 0 3  # 0.2 m/s forward for 3 s, then stop
nano$ python3 jetson/tools/base_console.py drive 0 1000 2 # turn in place at 1 rad/s
nano$ python3 jetson/tools/base_console.py estop          # flags bit0 set, wheels stop, pin 13 high
nano$ python3 jetson/tools/base_console.py clear
nano$ sudo systemctl start beni-ros
```

`flags` bits: 0 e-stop, 1 bumper L, 2 bumper R, 3 cliff, 4 charging, 5 watchdog.

| Check | Expect |
|---|---|
| `err` stays 0 | wiring and baud are fine (a growing count means noise or a swapped TX/RX) |
| `drive 200 0` gives v ≈ 200 | the PID is tracking; if v is negative, swap that motor's leads or the encoder A/B |
| Lift a wheel off the table edge | flags bit 3 (cliff) and a stop. With no ToF fitted, every reading is a cliff unless `ALLOW_MISSING_TOF 1` (bench only) |
| Kill `drive` mid-run (Ctrl-C) | the wheels stop within 300 ms (watchdog) |

Then run Phase 1 of [ACCEPTANCE.md](../ACCEPTANCE.md).
