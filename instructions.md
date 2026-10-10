# Building and running Beni: step-by-step instructions

This file takes you from an empty desk to a working robot in one sequence. Each step says what you are doing, why,
and the exact commands to type. The detailed guides in [docs/](docs/README.md) cover the same ground with more
background. The blueprint ([Beni_Robot_Engineering_Blueprint.md](Beni_Robot_Engineering_Blueprint.md)) explains the
design.

**How to read the commands**
- `pc$` means a terminal on your own computer.
- `nano$` means a terminal on the Jetson Nano (over SSH or a keyboard), logged in as your normal user, not `beni`.
- `kaggle>` means something you click on kaggle.com.
- Text in `<angle brackets>` is a value you replace, for example `<you>` is your Kaggle user name.
- Run every step in order. Every setup script can be safely run again: it skips work that is already done.

**The overall plan**

| Part | What happens | Where | Rough time |
|---|---|---|---|
| A | Check the code on your PC | PC | 15 min |
| B | Get the hardware and wire it | bench | a weekend |
| C | Prepare the Jetson Nano (Waveshare eMMC + SD card, headless) | Nano over SSH | 2–3 h (mostly waiting) |
| D | Make the AI models and TensorRT engines | PC or Kaggle, then Nano | 1–2 h |
| E | Build the face, the ROS container and the offline LLM | PC + Nano | 3 h (mostly waiting) |
| F | Flash the ESP32 motor board | PC | 30 min |
| G | Set up the Kaggle cloud brain | Linux PC or Kaggle + kaggle.com | 1 h |
| H | Configure and start everything | Nano | 15 min |
| I | Prove each phase works (acceptance tests) | Nano | ongoing |
| J | Daily use, updates and fixes | Nano | — |

---

## Part A: Check the code on your PC

You need Python 3.10 or newer. The tests use fake ("stub") AI models, so they need no GPU, no Jetson and no
internet.

**A1. Keep big caches off a small C: drive (Windows).** If C: is nearly full, send pip's cache and temp files to
another drive before you install anything:

```bat
pc$ mkdir D:\cache\pip D:\cache\tmp
pc$ setx PIP_CACHE_DIR D:\cache\pip
pc$ setx TEMP D:\cache\tmp
pc$ setx TMP D:\cache\tmp
```

Close and reopen the terminal so the new settings take effect.

**A2. Get the code and a virtual environment.** Put the repo and its venv on a drive with space:

```bash
pc$ git clone <your-repo-url> beni
pc$ cd beni
pc$ python -m venv .venv
pc$ . .venv/bin/activate              # Windows: .venv\Scripts\activate
pc$ pip install pytest pytest-asyncio "websockets>=14" openai msgpack numpy pyflakes
```

**A3. Run the tests and the linter.** All tests should pass, and pyflakes should print nothing:

```bash
pc$ make test        # same as: python -m pytest -q tests
pc$ make lint        # same as: python -m pyflakes shared kaggle jetson/agent jetson/tools jetson/vision jetson/engines tests
pc$ make e2e         # only the end-to-end brain test
```

On Windows without `make`, use the `python -m ...` forms shown in the comments.

**A4. (Optional) Run a fake brain on your PC.** This is useful later: you can point the robot at your PC to test the
robot side before the Kaggle brain is set up.

```bash
pc$ make brain-stub                                                 # Linux/macOS
pc$ set PYTHONPATH=shared;kaggle                                     # Windows, then:
pc$ python -m beni_brain.gateway --stub --host 127.0.0.1 --port 8765
pc$ curl http://127.0.0.1:8765/health                               # in a second terminal
```

To let the robot reach it, start it with `--host 0.0.0.0` and later set
`BENI_BRAIN_URL=ws://<pc-ip>:8765/ws` on the Nano (Part H).

---

## Part B: Get the hardware and wire it

The full parts list is in blueprint §2 and §4. The minimum you need is:
- a Jetson Nano 4 GB: yours is the Waveshare kit with the 16 GB eMMC module, running from a 128 GB microSD card;
- optionally, a USB 3 SSD of 128 GB or more as the data disk (recommended, see C3);
- two IMX219 cameras (one NoIR), an IR LED ring;
- an INMP441 I2S microphone (or two) and a MAX98357A I2S amplifier with a speaker;
- an HDMI LCD for the face, and optionally a touch panel (USB, or SPI XPT2046);
- a PCA9685 servo board for the head;
- an ESP32 dev board, a TB6612 motor driver, two encoder motors, an MPU6050, two VL53L0X cliff sensors, two HC-SR04
  ultrasonic sensors, two bumper switches, and optionally an INA219;
- a 3S battery with a buck converter;
- the RPLIDAR A1. It isn't needed until Phase 4, so you can buy it later.

**B1. Wire the ESP32 base.** Use this table, which comes from `firmware/base_esp32/main/config.h`:

| Function | ESP32 GPIO | Connects to |
|---|---|---|
| UART to the Jetson (921600 baud) | RX 16 / TX 17 | Jetson header pin 8 (TX) / pin 10 (RX), plus a shared ground |
| Left motor (TB6612 PWMA, AIN1, AIN2) | 25, 26, 27 | |
| Right motor (TB6612 PWMB, BIN1, BIN2) | 14, 12, 13 | |
| TB6612 standby | 33 | |
| Encoders: left A/B, right A/B | 34, 35, 36, 39 | add pull-ups if the encoders are open-collector |
| I2C SDA / SCL | 21 / 22 | MPU6050, both VL53L0X (XSHUT on 4 and 5), INA219 |
| Ultrasonic: left trig/echo, right trig/echo | 18/19, 23/15 | put a voltage divider on each 5 V echo line |
| Bumpers left / right | 0 / 3 | switches to ground |
| Battery voltage | 32 | 100 k / 22 k divider from the pack |
| E-stop signal out | 2 | Jetson header pin 13 |

Both boards use 3.3 V logic, so the UART and e-stop wires need no level shifter.

**B2. Wire the Jetson header.** These are the pins you'll switch on in step C8:
- I2S audio on pins 12, 35, 38 and 40 (amp and mic);
- PWM for the IR LED ring on pin 32;
- SPI for the touch panel on pins 19, 21, 23 and 24 (skip this if the panel is USB);
- the PCA9685 on I2C bus 1 (pins 3 and 5).

**B3. Wire the front-panel buttons.** Put a momentary switch across the power-button pins and another across the
reset pins, on the carrier's button header. It's J50 on NVIDIA's B01 board; on the Waveshare carrier, follow the
silkscreen (PWR / RST). Leave the auto-power-on pins open, so the Nano starts when the battery is
switched on.

**B4. Set the robot's measurements.** Before you build the firmware, edit `firmware/base_esp32/main/config.h`:
- `TICKS_PER_REV`, `WHEEL_RADIUS_M`, `TRACK_M` and `MAX_WHEEL_MPS` for your wheels;
- `BATT_CAL`, which is your multimeter's reading divided by the voltage the robot reports.

---

## Part C: Prepare the Jetson Nano (Waveshare eMMC board, OS on the SD card, headless)

**About your board.** Your Waveshare kit uses the production Nano module:
- It has **16 GB of eMMC** soldered on. The bootloader always lives there, and often the kernel and its
  `extlinux.conf` start-up file too.
- Your operating system now runs from the **128 GB microSD card**.

Two rules follow:
- **Store nothing on the eMMC.** It is too small, and it is what makes the board start.
- **Before changing pins or the kernel, run the check in step C7.** Depending on how the OS was moved to the SD card,
  the Nano may still read its start-up files from the eMMC. A pin change you make then needs copying to the eMMC
  before it takes effect.

"Headless" means no monitor, keyboard or desktop. You run everything over SSH from your PC. The HDMI screen will
still show Beni's face, because the face program drives the screen directly, without a desktop.

**C1. Get a network connection and log in over SSH.** An Ethernet cable is easiest. Find the Nano's address in your
router's device list, then from the PC:

```bash
pc$ ssh <your-user>@<nano-ip>
```

For Wi-Fi without a screen, use the Ethernet cable or the USB method below once, then connect Wi-Fi from the
command line:

```bash
nano$ nmcli device wifi list
nano$ sudo nmcli device wifi connect "<wifi-name>" password "<wifi-password>"
nano$ nmcli connection modify "<wifi-name>" connection.autoconnect yes
nano$ hostname -I                                   # the Nano's address(es)
```

**If you have no network at all:** connect a USB cable from the PC to the Nano's **micro-USB** port. The Nano then
appears as a network device on the PC at `192.168.55.1`:

```bash
pc$ ssh <your-user>@192.168.55.1
```

As a last resort, use a 3.3 V USB-serial adapter on the debug UART, on the header next to the module. Open it at
115200 baud, for example with PuTTY on Windows. This console also shows the boot menu.

**C2. Check what you're running.** All four checks must pass before you continue:

```bash
nano$ cat /etc/nv_tegra_release      # must say R32 (release), REVISION: 7.x   (= JetPack 4.6.x)
nano$ findmnt -no SOURCE /           # must say /dev/mmcblk1p1   (the SD card; mmcblk0 is the eMMC)
nano$ lsblk                          # mmcblk0 ~14.7G = eMMC, mmcblk1 ~119G = SD card
nano$ df -h /                        # Size should be ~115G or more
```

If `df -h /` shows only about 14G, the SD partition was copied at the eMMC's size. Grow it to fill the card:

```bash
nano$ sudo apt-get update && sudo apt-get install -y cloud-guest-utils
nano$ sudo growpart /dev/mmcblk1 1
nano$ sudo resize2fs /dev/mmcblk1p1
nano$ df -h /
```

If the release isn't R32.7.x, the rest of this guide won't work. You need to re-flash JetPack 4.6.x with
Waveshare's guide, which requires an Ubuntu 18.04 PC with NVIDIA SDK Manager.

**C3. Choose where Beni's data goes (`/ssd`).** Everything Beni stores lives under `/ssd`: models, engines,
recordings, memory, Docker images and swap. Pick one option.

*Option A: SD card only (simplest).* `/ssd` is just a folder on the SD card. The card will wear faster because of
the camera recordings, so use a good "High Endurance" card.

```bash
nano$ sudo mkdir -p /ssd
```

*Option B: a USB SSD as the data disk (recommended).* The OS stays on the SD card and the SSD is mounted at `/ssd`.
**This erases the SSD**, so use `lsblk` to make sure it really is `/dev/sda`:

```bash
nano$ lsblk
nano$ sudo parted -s /dev/sda mklabel gpt mkpart primary ext4 0% 100%
nano$ sudo mkfs.ext4 -L beni-ssd /dev/sda1
nano$ sudo mkdir -p /ssd
nano$ echo 'LABEL=beni-ssd /ssd ext4 defaults,noatime,nofail 0 2' | sudo tee -a /etc/fstab
nano$ sudo mount /ssd && df -h /ssd
```

Don't use `make ssd-root` on this board. That moves the OS itself, which you've already done with the SD card.

**C4. Get the code and install the Beni user and services.**

```bash
nano$ sudo apt-get update && sudo apt-get install -y git make
nano$ git clone <your-repo-url> ~/beni && cd ~/beni
nano$ sudo make jetson-install
```

This step:
- creates the `beni` user that the services run as;
- links `/opt/beni` to `~/beni`;
- creates the folders under `/ssd`;
- copies the settings template to `/etc/beni/beni.env`;
- installs and enables the systemd services.

**C5. (Optional) Make a Tailscale key for remote access from your PC.** Tailscale lets you reach the robot from your PC anywhere as `beni-jetson` (the robot and cloud brain connect over the Cloudflare Worker Relay, Part G):
1. Make a free account at tailscale.com.
2. In the admin console, create an auth key that is **reusable** and **pre-approved**, with the tag `tag:beni-jetson`.
3. In Access Controls, paste this:

```json
"tagOwners": {"tag:beni-jetson": ["autogroup:admin"]},
"acls": [{"action": "accept", "src": ["autogroup:admin"], "dst": ["tag:beni-jetson:22,8889"]}]
```

4. Install Tailscale on your PC as well and log in. You can then reach the robot from anywhere as `beni-jetson`. (On local home Wi-Fi, you can also connect directly via the Nano's local IP address).

**C6. Install everything on the Nano.** This one command runs the setup scripts in order, and each script installs
only what is missing. The list of what gets installed is in the appendix at the end of this file.
- `00_jetpack.sh`: CUDA, TensorRT, the Multimedia API, Docker with GPU support, DeepStream 6.0.1, pyds and jtop. An
  eMMC board flashed by Waveshare's method usually lacks these.
- `01_system_tune.sh`: switches to headless mode, sets maximum performance, swap, gcc-9, the watchdog and system
  packages.
- `03_docker.sh`: points Docker at `/ssd/docker` and downloads the ROS base image (~3 GB).
- `04_tailscale.sh`: joins your tailnet as `beni-jetson`.
- `05_py38_venv.sh`: builds the Python 3.8 environment for the agent (30–60 min).
- `08_mediamtx.sh`: the video server for watching the cameras.
- `10_deepstream_yolo.sh`: builds the YOLO parser for DeepStream.

Run it inside `screen`, so an SSH disconnect doesn't kill it. If you get disconnected, log back in and run
`screen -r` to get the session back.

```bash
nano$ sudo apt-get install -y screen && screen
nano$ cd ~/beni
nano$ TS_AUTHKEY=tskey-auth-<your-key> make jetson-setup          # 1–2 h
nano$ sudo reboot
```

After the reboot, log in again and check the install:

```bash
nano$ ssh <your-user>@beni-jetson                       # over Tailscale, from now on
nano$ systemctl get-default                             # multi-user.target  (= headless)
nano$ ls /usr/src/tensorrt/bin/trtexec /opt/nvidia/deepstream/deepstream-6.0
nano$ ls /opt/beni/.venv38/bin/python
nano$ docker info | grep -E "Docker Root Dir|Default Runtime"    # /ssd/docker, nvidia
nano$ free -m                                           # "available" ~3300 MB or more with nothing running
nano$ sudo jtop                                         # live CPU/GPU/temperature view; q to quit
```

`01_system_tune.sh` turns off the `avahi` service, so `<name>.local` addresses stop working. Use the Tailscale name
`beni-jetson` or the IP address instead.

**C7. Find out which `/boot` your Nano starts from (eMMC boards only).** This takes two runs of the check, with a
reboot between them:

```bash
nano$ cd ~/beni && make emmc-boot-check      # 1st run: marks the SD card's start-up file
nano$ sudo reboot
nano$ cd ~/beni && make emmc-boot-check      # 2nd run: tells you the answer
```

- **"OK: this rootfs's own /boot is what boots":** nothing special to do. Skip the sync steps below.
- **"The eMMC's /boot is what boots":** after **every** pin change (step C8) and every kernel update (`apt upgrade`
  that touches `nvidia-l4t-kernel`), run `make emmc-boot-sync` before rebooting. The sync copies the SD card's
  `/boot` to the eMMC, keeps the OS on the SD card, and adds an `emmc-rootfs` recovery entry to the serial-console
  boot menu.

**C8. Switch on the header pins for audio, the IR LEDs and SPI touch.** The jetson-io tool is a text menu, which
works fine over SSH. Make the terminal window at least 80×25 first.

```bash
nano$ sudo /opt/nvidia/jetson-io/jetson-io.py
```

In the menu:
1. Choose "Configure Jetson 40pin Header", then "Configure header pins manually".
2. Tick `i2s4`, `pwm0` and `spi1`. Leave `spi1` off if the touch panel is USB.
3. Choose "Back", then "Save pin changes".
4. If C7 said "eMMC's /boot", choose "Save and exit without rebooting", then run the sync:

```bash
nano$ cd ~/beni && make emmc-boot-sync
nano$ sudo reboot
```

5. Otherwise choose "Save and reboot to reconfigure pins".

After the reboot, check that the sound card exists. If it doesn't, the pin change didn't reach the eMMC: run C7
again.

```bash
nano$ aplay -l | grep tegrasndt210ref
```

**C9. Useful headless commands**

```bash
nano$ sudo shutdown -h now                        # always shut down like this before cutting power
nano$ sudo reboot
nano$ sudo systemctl set-default graphical.target # only if you ever want the desktop back (costs ~0.7 GB RAM)
nano$ sudo nmcli device wifi connect "<wifi-name>" password "<wifi-password>"   # add another Wi-Fi network
nano$ df -h / /ssd                                # disk space
nano$ sudo tegrastats                             # one-line live hardware stats; Ctrl-C to stop
```

---

## Part D: Make the AI models and TensorRT engines

The detector models are exported on a **PC or a Kaggle notebook**, because the export tools need a newer Python than
the Nano has. The TensorRT engines are then built **on the Nano**, because engines only work on the GPU they were
built on.

> The export downloads PyTorch (~2 GB). If your PC's system drive is small, do Part D in a Kaggle notebook (put `!`
> before each command), then download the files from `/kaggle/working`.

**D1. Download the CPU models on the Nano.** These are the wake word, voice detection, offline speech recognition,
offline voice, speaker ID, the memory embedder and the filler sounds:

```bash
nano$ cd ~/beni && make models
nano$ ls /ssd/beni/models
```

**D2. Export the object detectors (PC or Kaggle).** The second export uses the same DeepStream-Yolo version the Nano
built in C6:

```bash
pc$ python -m venv ~/yolo && . ~/yolo/bin/activate
pc$ pip install -U ultralytics onnx onnxslim onnxsim
pc$ mkdir -p ~/onnx && cd ~/onnx
pc$ python ~/beni/jetson/engines/export_yolo26n.py                       # -> yolo26n_512x288_b2.onnx
pc$ export DSY_REF=$(ssh <you>@beni-jetson cat beni/third_party/DeepStream-Yolo.sha)
pc$ bash ~/beni/jetson/engines/export_deepstream_yolo.sh                 # -> yolo26n_ds_..., yolov8n_..., labels.txt
```

Optional extras for "light mode":

```bash
pc$ python ~/beni/jetson/engines/export_yolo26n.py yolo26n.pt 224 416
pc$ SIZE="224 416" bash ~/beni/jetson/engines/export_deepstream_yolo.sh
```

**D3. Get the face, re-identification and pose models.** These come from third-party projects. Each must have a
fixed batch size that matches its file name:

| File | Where it comes from |
|---|---|
| `scrfd_500m_320_b4.onnx` | insightface `scrfd_500m_bnkps`, exported at 320×320 with landmarks (batched outputs) |
| `mobilefacenet_112_b8.onnx` | insightface `buffalo_s` → `w600k_mbf.onnx` |
| `osnet_x0_25_256x128_b4.onnx` | torchreid `osnet_x0_25` exported with `torch.onnx.export` (opset 12) |
| `yolov8n_face_160_b8.onnx` | any YOLOv8n-face export (only the DeepStream path uses it) |
| `trtpose_r18_224_b4.onnx` | `python jetson/engines/export_trtpose.py resnet18_baseline_att_224x224_A_epoch_249.pth` |

If a model has a flexible batch size, fix it with this snippet, for example to batch 4:

```bash
pc$ python - scrfd_500m_dyn.onnx scrfd_500m_320_b4.onnx 4 <<'EOF'
import sys, onnx
from onnxsim import simplify
src, dst, b = sys.argv[1], sys.argv[2], int(sys.argv[3])
m = onnx.load(src)
for i in m.graph.input:
    i.type.tensor_type.shape.dim[0].dim_value = b
m, ok = simplify(m)
assert ok
onnx.save(m, dst)
EOF
```

A missing model isn't fatal. That feature simply turns off: with no OSNet there's no person re-identification, and
with no TRT-Pose there are no gestures.

**D4. Copy the models to the Nano and build the engines.** Each engine takes 5–15 minutes:

```bash
pc$   scp ~/onnx/*.onnx ~/onnx/labels.txt <you>@beni-jetson:/tmp/
nano$ sudo install -o beni -g beni -m 644 /tmp/*.onnx /tmp/labels.txt /ssd/beni/models/
nano$ cd ~/beni && make engines
nano$ ls -la /ssd/beni/engines/*.engine
nano$ grep -l FAILED /ssd/beni/engines/*.log                 # should print nothing
nano$ sudo -u beni bash jetson/engines/build_all.sh --bench  # speed of each engine
```

If YOLO26n fails to build:
1. Re-export it with `OPSET=13 python export_yolo26n.py`.
2. Rebuild with `sudo -u beni env FORCE=1 bash jetson/engines/build_all.sh`.
3. If it still fails, set `BENI_DETECTOR=yolov8n` in `/etc/beni/beni.env`.

---

## Part E: Build the face, the ROS container and the offline LLM

**E1. The face display.** Rendered clips are already in `jetson/face/assets/clips/`. Re-render them only if you
change the face design, which needs numpy plus x264 or ffmpeg on the PC:

```bash
pc$   make face-clips
pc$   rsync -a jetson/face/assets/clips/ <you>@beni-jetson:beni/jetson/face/assets/clips/
```

Build the face program and install the clips:

```bash
nano$ cd ~/beni && make face
```

**E2. The ROS 2 container.** This runs the motor driver bridge, the head servos, SLAM and navigation. It takes about
2 hours the first time and about 5 minutes after that. The image lives on the SSD.

```bash
nano$ cd ~/beni && make ros-image
nano$ docker images beni-ros
```

**E3. The offline LLM.** This builds llama.cpp (~40 min) and downloads Qwen2.5-0.5B (~400 MB). Beni uses it to
answer when the cloud brain is off.

```bash
nano$ cd ~/beni && make llama
```

**E4. (Phase 3 and later) The fast C++ vision core.** Until Phase 3 the DeepStream path (`beni-vision`) runs. When
you are ready, build vision_core and switch to it:

```bash
nano$ cd ~/beni && make vision-core
nano$ sudo systemctl disable --now beni-vision
nano$ sudo systemctl enable --now beni-vision-core
nano$ journalctl -u beni-vision-core -f              # look for "[infer] detector /ssd/beni/engines/..."
```

**E5. (Recommended, Phase 5) The encrypted memory vault.** This keeps memories and face data on an encrypted SSD
file, with the key on the microSD card. It asks you for a recovery passphrase; write it down.

```bash
nano$ cd ~/beni && sudo make vault
```

---

## Part F: Flash the ESP32 motor board

> The toolchain is 1–2 GB. If your PC's system drive is small, point it at another drive **first**:
> Windows `setx PLATFORMIO_CORE_DIR D:\pio`, Linux `export PLATFORMIO_CORE_DIR=/data/pio`.

**F1. Build and flash with PlatformIO.**

```bash
pc$ pip install -U platformio
pc$ cd beni/firmware/base_esp32
pc$ pio run                                           # first build downloads the toolchain
pc$ pio run -t upload --upload-port /dev/ttyUSB0      # Windows: --upload-port COM5
pc$ pio device monitor                                # boot log over USB; Ctrl-C to quit
```

If you prefer plain ESP-IDF 5.3.1:

```bash
pc$ . ~/esp/esp-idf-v5.3.1/export.sh
pc$ cd beni/firmware/base_esp32 && idf.py set-target esp32 && idf.py build
pc$ idf.py -p /dev/ttyUSB0 flash monitor
```

**F2. Bench-test from the Nano.** Prop the robot up so the wheels are off the ground first.

```bash
nano$ sudo systemctl stop beni-ros                         # frees the serial port
nano$ cd ~/beni
nano$ python3 jetson/tools/base_console.py monitor         # live readings; "err" must stay 0
nano$ python3 jetson/tools/base_console.py drive 200 0 3   # forward at 0.2 m/s for 3 s
nano$ python3 jetson/tools/base_console.py drive 0 1000 2  # spin on the spot
nano$ python3 jetson/tools/base_console.py estop           # wheels stop, e-stop flag set
nano$ python3 jetson/tools/base_console.py clear
nano$ sudo systemctl start beni-ros
```

| Problem | Fix |
|---|---|
| A wheel spins the wrong way | swap that motor's two wires, or its encoder A/B wires |
| `err` keeps counting | TX and RX are swapped, or the grounds aren't joined |
| Every reading says "cliff" | a ToF sensor is missing or miswired |

---

## Part G: Set up the Kaggle cloud brain

The brain is a private Kaggle notebook running on two free T4 GPUs:
- GPU 0 runs the main language-and-vision model.
- GPU 1 runs speech-to-text, the voice and the object finder.

You don't start it by hand day to day. The robot starts it in the morning or when you say the wake word, and stops
it when idle.

**G1. Kaggle account and API key.**
1. On kaggle.com, verify your phone number. GPUs and internet access need it.
2. Go to Settings → API → **Create New Token**. This downloads `kaggle.json`.
3. Put the key on the PC:

```bash
pc$ mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
pc$ pip install "kaggle>=1.6,<1.7"
pc$ kaggle datasets list -m                 # works = the key is fine
```

**G2. Set up the free Cloudflare Worker Relay.** Tailscale gets detected and killed within 15 seconds by Kaggle's container supervisor. Instead, Beni uses a **free, permanent Cloudflare Worker WebSocket Relay** (`cloudflare/worker.js`):
1. In `cloudflare/`, run `npx wrangler login` followed by `npx wrangler deploy`.
2. Wrangler automatically runs the SQLite Durable Object migration and links the `RELAY` binding.
3. Your permanent relay URL is: `wss://beni-relay.<your-subdomain>.workers.dev` (e.g. `wss://beni-relay.impjrimpjr.workers.dev`).

**G3. Pick a shared password.** The robot and the brain must use the same token:

```bash
pc$ openssl rand -hex 24
```

Keep this string. You'll use it in G4 and in Part H.

**G4. Add Kaggle Secrets.** In `kaggle>` Code → New Notebook → Add-ons → Secrets, add:

| Secret | Value |
|---|---|
| `BENI_TOKEN` | the string from G3 (required) |
| `BENI_RELAY_URL` | `wss://beni-relay.<your-subdomain>.workers.dev` from G2 (required) |
| `HF_TOKEN` | a Hugging Face read token (recommended, for faster model downloads) |
| `BENI_HOME_CITY`, `BENI_TZ` | your city and time zone, e.g. `Colombo`, `Asia/Colombo` (optional) |

The optional extras are `BENI_TTS`, `BENI_VISION`, `BENI_RERANKER`, `BENI_LLM_REPO`, `BENI_LORA`,
`BENI_ADAPTER_REPO`, `BENI_PERSONA` and `CF_TUNNEL_TOKEN`. They are explained in
[docs/04_KAGGLE_BRAIN.md](docs/04_KAGGLE_BRAIN.md#43-kaggle-secrets).

**G5. Build and upload the wheelhouse.** The wheelhouse is a bundle of every Python package the brain needs, so the
notebook starts quickly. It needs **Linux x86_64 with Python 3.12** (or Docker), and it downloads about 3 GB. Don't
run it on a nearly full drive: a spare Linux machine or a Kaggle notebook is ideal.

```bash
pc$ cd beni && make wheelhouse-push            # creates the dataset <you>/beni-wheelhouse
```

Do this again whenever `shared/`, `kaggle/beni_brain/` or `kaggle/wheelhouse/requirements-*.txt` change.

**G6. First run by hand.**

```bash
pc$ cd beni
pc$ make kaggle-push KAGGLE_KERNEL=<you>/beni-brain
pc$ kaggle kernels status <you>/beni-brain       # queued, then running
```

Next, attach the secrets to the notebook. This is needed once only:
1. On kaggle.com, open the `beni-brain` notebook and choose Edit → Add-ons → Secrets.
2. Tick every secret from G4.
3. Choose Save Version.

Watch the notebook's log. After 10–15 minutes it prints `brain READY after N s`.

**G7. (Optional) Give Beni a cloned voice.**
1. Record 3–10 seconds of clean speech as `beni_voice.wav`.
2. Upload it as a private dataset, `<you>/beni-voice`.
3. Add `"<you>/beni-voice"` to `dataset_sources` in `kaggle/kernel-metadata.json`.
4. Set the secret `BENI_TTS=cosyvoice`.

---

## Part H: Configure and start everything

**H1. Fill in the robot's settings.**

```bash
nano$ sudoedit /etc/beni/beni.env
```

| Setting | What to put |
|---|---|
| `BENI_TOKEN` | the same string as the Kaggle secret (G3) |
| `BENI_BRAIN_URL` | `wss://beni-relay.<your-subdomain>.workers.dev/robot` (or `ws://<pc-ip>:8765/ws` to use the PC stub from A4) |
| `KAGGLE_KERNEL` | `<you>/beni-brain` |
| `HF_TOKEN`, `HF_BACKUP_REPO` | optional nightly memory backup to a private Hugging Face dataset |
| `BENI_DETECTOR` | `yolo26n` (default) or `yolov8n` |
| `BENI_AWAKE_HOURS` | when to keep the brain running, e.g. `07:00-12:00,16:00-22:30` |
| `BENI_WEEKLY_BUDGET_H` | GPU hours per week (default 28; Kaggle gives about 30) |
| `BENI_QUIET_HOURS` | when Beni shouldn't start conversations, e.g. `22:00-07:00` |
| `BENI_PHONES` | `Name=AA:BB:CC:DD:EE:FF` for "owner is home" detection (pair the phone first, see H1a) |
| `BENI_GPIO_MUTE` | sysfs number of a mic-mute switch pin, e.g. `194` for pin 15 |
| `BENI_TOUCH` | `auto`, `off` or `/dev/input/eventN` |

**H1a. (Optional) Pair a phone for "owner is home".** Make the phone discoverable (open its Bluetooth settings),
then on the Nano:

```bash
nano$ bluetoothctl
[bluetooth]# power on
[bluetooth]# agent on
[bluetooth]# scan on                 # wait until your phone's name and address appear
[bluetooth]# pair AA:BB:CC:DD:EE:FF  # accept on the phone
[bluetooth]# trust AA:BB:CC:DD:EE:FF
[bluetooth]# scan off
[bluetooth]# exit
```

Put that address in `BENI_PHONES` in step H1.

**H2. Give the robot the Kaggle key** so it can start and stop the brain itself:

```bash
pc$   scp ~/.kaggle/kaggle.json <your-user>@beni-jetson:~/         # Windows: scp %USERPROFILE%\.kaggle\kaggle.json ...
nano$ sudo install -d -o beni -g beni -m 700 /home/beni/.kaggle
nano$ sudo install -o beni -g beni -m 600 ~/kaggle.json /home/beni/.kaggle/kaggle.json
nano$ rm ~/kaggle.json
```

**H3. Start the services and check them.**

```bash
nano$ sudo systemctl restart beni-audio beni-vision beni-face beni-ros
nano$ sudo systemctl start beni-agent beni-sched
nano$ cd ~/beni && make status                       # every unit should be "active (running)"
nano$ journalctl -u beni-agent -f                    # watch for "link up"
```

**H4. Talk to it.** When the brain is up, check its status, then say the wake word ("Hey Beni") and ask something:

```bash
curl -s https://beni-relay.<your-subdomain>.workers.dev/health
# {"status": "ok", "brain_connected": true, "robot_connected": true, ...}
nano$ journalctl -u beni-agent -f                    # watch for "brain online via ..."
```

**H5. Watch the cameras.** On any device on your tailnet, open `http://<jetson-tailscale-ip>:8889/teleop`.

---

## Part I: Prove each phase works (acceptance tests)

Go through [ACCEPTANCE.md](ACCEPTANCE.md) one phase at a time, and don't move on until a phase passes. These are the
commands used most:

```bash
nano$ grep MemAvailable /proc/meminfo                              # Phase 0: at least 3.3 GB at idle
nano$ bash jetson/tools/bench/cams.sh 600                          # Phase 0: both cameras 59–60 fps for 10 min
nano$ python3 jetson/tools/base_console.py drive 200 0 10          # Phase 1: drive 2 m, measure with a tape
nano$ ls /ssd/beni/rec | wc -l                                     # Phase 1: recordings, 12 per camera per hour
nano$ python3 jetson/tools/voice_latency_report.py /ssd/beni/logs/turns.jsonl   # Phase 2: p50 under 2.0 s
nano$ journalctl -u beni-audio --since "1 hour ago" | grep -i xrun # Phase 2: should be empty
nano$ cd /opt/beni && sudo -u beni make bench                      # Phase 3: CPU/RAM/GPU per service
nano$ python3 jetson/tools/tegrastats_logger.py /ssd/beni/logs/tegra.csv 500   # GPU/thermal over time
nano$ journalctl -k | grep -i oom; systemctl --failed              # Phase 6: 24 h run, both empty
```

**Phase 4 (space) needs the RPLIDAR A1.** Plug it in and restart `beni-ros`. Then drive the robot around the house
with teleop to build the map:

```bash
nano$ sudo systemctl restart beni-ros
nano$ cd /opt/beni && sudo -u beni python3 jetson/tools/teleop.py
```

Teleop keys: `w`/`s`/`a`/`d` or the arrows to drive, `space` to stop, and `+`/`-` for speed. After that, name
places by voice ("this is the kitchen").

**Phase 6 (docking):** record docking demos, then turn them into a training dataset on a PC or Kaggle:

```bash
nano$ cd /opt/beni && sudo -u beni python3 jetson/tools/teleop.py --record "dock at the charger"   # r start/end, x discard
pc$   rsync -a <you>@beni-jetson:/ssd/beni/lerobot/raw/ raw/ && rsync -a <you>@beni-jetson:/ssd/beni/rec/ rec/
pc$   python jetson/tools/lerobot_export.py raw rec beni_demos
```

**Things you calibrate by hand:**
- the gesture thresholds in `perception/gestures.py`;
- the NoIR white balance (`CamCfg::wb`) and the motion-vector scale (`Encoder::mv_scale`) in vision_core.

---

## Part J: Daily use, updates and fixes

**Everyday commands**

```bash
nano$ cd ~/beni && make status                                     # is everything running?
nano$ journalctl -u beni-agent -f                                  # live log of any service
nano$ sudo systemctl restart beni-agent beni-sched                 # after editing /etc/beni/beni.env
nano$ sudo systemctl restart beni-audio beni-vision beni-face beni-ros beni-agent beni-sched   # restart everything
nano$ sudo -u beni /opt/beni/.venv38/bin/kaggle kernels status <you>/beni-brain              # is the brain up?
```

**Privacy:** say "privacy mode" or "close your eyes" to switch the cameras off. Say "open your eyes" to switch them
back on. "Not now" pauses Beni's own conversation-starting for 30 minutes.

**Updating the code**

```bash
nano$ cd ~/beni && git pull
nano$ sudo apt-get update && sudo apt-get upgrade                 # OS updates (JetPack 4.6.6 is the last 4.x)
nano$ make emmc-boot-sync                                         # only if C7 said "eMMC's /boot" and the kernel was updated
nano$ sudo make jetson-install                                     # only if service files changed
nano$ .venv38/bin/pip install -e shared[memory] -e jetson/agent    # only if requirements changed
nano$ make face                                                    # only if jetson/face changed
nano$ make vision-core                                             # only if jetson/vision_core changed
nano$ sudo systemctl restart beni-agent beni-sched
pc$   make wheelhouse-push                                         # only if shared/ or kaggle/ changed
```

**Switching the detector**

```bash
nano$ sudoedit /etc/beni/beni.env                  # BENI_DETECTOR=yolo26n | yolov8n | yolo26n_416 | yolov8n_416
nano$ sudo systemctl restart beni-vision-core      # or beni-vision before Phase 3
```

**Backing up to a NAS over the Ethernet cable**

```bash
nano$ bash jetson/tools/eth_sync.sh <you>@<nas>:/tank/beni --dry-run
nano$ bash jetson/tools/eth_sync.sh <you>@<nas>:/tank/beni
```

**Common problems**

| Symptom | What to do |
|---|---|
| A service won't start | `journalctl -u <unit> -b` and read the last lines |
| `No module named zmq` in beni-vision | `bash jetson/setup/05_py38_venv.sh` |
| "no detector" or engine errors | rebuild with `make engines`; check `/ssd/beni/engines/*.log` |
| The engine build was killed (out of memory) | `sudo systemctl stop beni-vision beni-vision-core`, then `make engines` |
| Black cameras | `sudo systemctl restart nvargus-daemon`, then restart the vision unit; check the ribbon cables |
| No sound | `aplay -l` must show `tegrasndt210ref`; redo steps C7–C8 if not |
| Blank face | `sudo systemctl set-default multi-user.target`, then reboot; check `/ssd/face` has clips |
| "Cloud offline" | check G6 and H4; meanwhile the offline LLM answers |
| `kaggle push failed` | `/home/beni/.kaggle/kaggle.json` exists with mode 600, and `KAGGLE_KERNEL` is set |
| Robot won't move | is `beni-ros` running? `base_console.py monitor` shows the e-stop, cliff or watchdog flags |
| Running hot | Beni slows down above 80 °C; check the fan |
| The Nano won't boot after a sync | on the serial console (115200 baud), pick `emmc-rootfs` in the boot menu; the old eMMC start-up file is kept as `/boot/extlinux/extlinux.conf.beni-<date>` on the eMMC |

Deeper troubleshooting is in [docs/06_OPERATION.md](docs/06_OPERATION.md#68-troubleshooting) and
[docs/04_KAGGLE_BRAIN.md](docs/04_KAGGLE_BRAIN.md#48-troubleshooting).

---

## Appendix: everything that gets installed on the Jetson Nano

You don't install these by hand. The commands in Parts C–E do it, and every script skips what is already there.
This list is so you know what ends up on the Nano, and where. The total is about 20–25 GB, which fits easily on the
128 GB card.

| Installed by | What | Where |
|---|---|---|
| `make jetson-setup` → `00_jetpack.sh` | git, make, curl, rsync, pip, python3-gi, bluez, parted, cryptsetup, nano | apt |
| | **JetPack components** (`nvidia-jetpack`): CUDA 10.2, cuDNN 8.2, TensorRT 8.2 + `trtexec`, VPI, Jetson Multimedia API | `/usr/local/cuda`, `/usr/src/tensorrt`, `/usr/src/jetson_multimedia_api` |
| | Docker + NVIDIA container runtime | apt |
| | **DeepStream 6.0.1** + GStreamer plugins (good, bad, ugly, libav, rtsp-server) | `/opt/nvidia/deepstream/deepstream-6.0` |
| | **pyds 1.1.1** (DeepStream Python bindings, host Python 3.6) | system Python |
| | jetson-stats (`jtop`) | system Python |
| `01_system_tune.sh` | Python 3.8 (+venv, dev), libopus, ZeroMQ, SQLite dev, gst-rtsp-server, alsa-utils, i2c-tools, smem, cmake, pkg-config | apt |
| | **gcc-9 / g++-9** (for llama.cpp) | ubuntu-toolchain-r PPA |
| | headless boot, MAXN power mode, locked clocks, IRQ pinning, 4 GB swapfile, hardware watchdog, Wi-Fi power-save off, the UART freed for the ESP32 | systemd, `/etc` |
| `03_docker.sh` | Docker data-root on `/ssd/docker`; ROS base image `dustynv/ros:humble-ros-base-l4t-r32.7.1` | `/ssd/docker` |
| `04_tailscale.sh` | Tailscale | apt |
| `05_py38_venv.sh` | Agent venv: numpy, pyzmq, msgpack, websockets, onnxruntime, tokenizers, sherpa-onnx, kaggle, huggingface_hub, beni_common, beni_agent | `/opt/beni/.venv38` |
| | Host Python 3.6: numpy 1.19.5, pyzmq, msgpack, pyserial, smbus2 | system Python |
| `08_mediamtx.sh` | MediaMTX v1.9.3 video server + service | `/usr/local/bin/mediamtx` |
| `10_deepstream_yolo.sh` | DeepStream-Yolo parser library | `~/beni/third_party/DeepStream-Yolo` |
| `make models` | wake word, voice detector, offline speech recognition (Moonshine), offline voice (Piper), speaker ID (CAM++), bge-small embedder, filler clips | `/ssd/beni/models` |
| `make llama` | llama.cpp `llama-server` + Qwen2.5-0.5B GGUF | `~/beni/jetson/llm`, `/ssd/beni/models` |
| `make engines` | TensorRT engines from your ONNX files | `/ssd/beni/engines` |
| `make face` | `beni_face` program + expression clips | `~/beni/jetson/face/build`, `/ssd/face` |
| `make ros-image` | ROS 2 Humble image with Nav2, slam_toolbox, sllidar and the Beni packages | `/ssd/docker` |
| `make vision-core` | the C++ vision program | `~/beni/jetson/vision_core/build` |
| `make vault` | encrypted vault file; its key goes on the eMMC partition (`/dev/mmcblk0p1`), separate from the data | `/ssd/beni-vault.img` |
| `sudo make jetson-install` | `beni` user, systemd units, sudoers, tmpfiles, `/etc/beni/beni.env` | `/etc` |

**If `00_jetpack.sh` stops at DeepStream**, NVIDIA's apt server didn't offer the package. Download
`deepstream-6.0_6.0.1-1_arm64.deb` from developer.nvidia.com (DeepStream 6.0.1 for Jetson), copy it to the Nano,
then:

```bash
nano$ sudo apt-get install -y ./deepstream-6.0_6.0.1-1_arm64.deb
nano$ cd ~/beni && make jetson-setup                # carries on from where it stopped
```
