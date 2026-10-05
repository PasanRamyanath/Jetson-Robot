# 2. Jetson Nano setup (JetPack 4.6.6 / L4T 32.7.6)

From a blank microSD card to all Beni services installed. **eMMC boards** (e.g. the Waveshare kit) already running
from an SD card: follow Part C of [instructions.md](../instructions.md#part-c-prepare-the-jetson-nano-waveshare-emmc-board-os-on-the-sd-card-headless)
instead of 2.1–2.2, and run `make emmc-boot-check` before jetson-io (2.5). Run the steps in order. Every script can be re-run
safely: each one skips work that is already done.

You need:
- a Nano 4 GB;
- a USB 3 SSD (≥ 128 GB);
- a 64 GB microSD card;
- the two IMX219 cameras, the I2S amp and mic, and the HDMI LCD (§2);
- a network cable or Wi-Fi.

A USB-serial adapter on the debug UART (the J50 button header on the B01 carrier) is handy if the SSD boot ever
fails. Header pins 8/10 are not the console: they are `/dev/ttyTHS1`, which the ESP32 base uses.

Front-panel buttons (§4 item 55; the blueprint's "J40" is the A02 name) wire straight to that header. Follow the
silkscreen: a momentary switch across the power-button pins and another across system reset. No software is needed.
Leave the auto-power-on pins open, so the Nano boots as soon as the pack is switched on.

## 2.1 Flash JetPack and do first boot

1. On the PC, write the **JetPack 4.6.6** SD-card image (`jetson-nano-jp466-sd-card-image.zip`) with Etcher.
2. Boot the Nano and finish oem-config. Create your normal user (called `you` below). Do **not** name it `beni`,
   because `make jetson-install` creates that account for the services.
3. Update and install the basics:

```bash
nano$ sudo apt-get update && sudo apt-get install -y git make curl rsync
nano$ cat /etc/nv_tegra_release           # expect: R32 (release), REVISION: 7.6
```

## 2.2 Move the rootfs to the SSD (§2.5 step 9)

Boot stays on the microSD card; the rootfs runs from the SSD. The script **never formats**, so you partition the
disk yourself. This erases the SSD:

```bash
nano$ lsblk                                    # find the SSD, e.g. /dev/sda
nano$ sudo parted -s /dev/sda mklabel gpt mkpart primary ext4 0% 100%
nano$ sudo mkfs.ext4 -L beni-ssd /dev/sda1
nano$ git clone <your-repo-url> ~/beni && cd ~/beni
nano$ sudo make ssd-root PART=/dev/sda1 CONFIRM=1     # rsyncs / to the SSD, adds a "recovery" boot entry
nano$ sudo reboot
nano$ findmnt -no SOURCE /                           # after the reboot: /dev/sda1
```

If the SSD doesn't boot, choose the `recovery` entry on the serial console. It boots the untouched SD rootfs.
`/ssd` is now a plain directory on the SSD root, and all data goes under it.

## 2.3 Install the Beni user, units and config

```bash
nano$ cd ~/beni
nano$ sudo make jetson-install
```

This does the following:
- creates user `beni`, in the audio, video, dialout, i2c and gpio groups;
- links `/opt/beni` to `~/beni`;
- makes `/ssd/beni/{models,engines,logs}`, `/ssd/maps` and `/ssd/face`;
- copies `beni.env.example` to `/etc/beni/beni.env` (mode 0600);
- installs the tmpfiles and sudoers entries (`beni` may start/stop `beni-llm` and the vision units, and set nvpmodel) and all
  the systemd units;
- enables audio, vision, face, ros, agent and sched.

The services run as `beni` from `/opt/beni`, so your home must be readable. On 18.04 it is (0755). If you tightened
it, run `chmod 755 ~`.

## 2.4 System tuning, Docker, Tailscale, the py3.8 venv, MediaMTX and DeepStream-Yolo

First create a Tailscale auth key. In the Tailscale admin console, create a **reusable, pre-approved** key tagged
`tag:beni-jetson`. Then add this to the ACL:

```json
"tagOwners": {"tag:beni-jetson": ["autogroup:admin"], "tag:beni-brain": ["autogroup:admin"]},
"acls": [{"action": "accept", "src": ["tag:beni-jetson"], "dst": ["tag:beni-brain:8765"]},
         {"action": "accept", "src": ["autogroup:admin"], "dst": ["tag:beni-jetson:22,8889"]}]
```

The second rule lets your own devices SSH in and watch the teleop stream (WebRTC on :8889).

```bash
nano$ cd ~/beni
nano$ TS_AUTHKEY=tskey-auth-... make jetson-setup
nano$ sudo reboot
```

| Script | What it does |
|---|---|
| `00_jetpack.sh` | installs what an eMMC flash lacks: `nvidia-jetpack` (CUDA, TensorRT, MMAPI), nvidia-docker, DeepStream 6.0.1, pyds 1.1.1, jtop (skips what is there) |
| `01_system_tune.sh` | nvpmodel MAXN and jetson_clocks at boot; zram plus a 4 GB swapfile on the SSD; sysctl; apt packages (python3.8, opus, zmq and more); CPU and IRQ pinning; gcc-9; the hardware watchdog; chowns data dirs to `beni` |
| `03_docker.sh` | Docker data-root at `/ssd/docker`, with the NVIDIA runtime as the default |
| `04_tailscale.sh` | joins the tailnet as `beni-jetson` with `tag:beni-jetson`; prints the IP (skipped if already joined) |
| `05_py38_venv.sh` | `/opt/beni/.venv38` with `jetson/agent/requirements.txt` (sherpa-onnx builds from source if no wheel fits) |
| `08_mediamtx.sh` | MediaMTX v1.9.3 plus `mediamtx.service` (WebRTC teleop on :8889, RTSP on 127.0.0.1:8554) |
| `10_deepstream_yolo.sh` | clones DeepStream-Yolo into `third_party/`, records its commit in `third_party/DeepStream-Yolo.sha`, and builds `libnvdsinfer_custom_impl_Yolo.so` with `CUDA_VER=10.2` |

The venv build is the slow part (30–60 min). Check it with `ls /opt/beni/.venv38/bin/python`.

## 2.5 Enable the 40-pin functions (i2s4, pwm0, spi1)

jetson-io is interactive, so it runs once by hand:

```bash
nano$ sudo /opt/nvidia/jetson-io/jetson-io.py
#  Configure Jetson 40pin Header -> Configure header pins manually ->
#    [*] i2s4   (pins 12/35/38/40: MAX98357A amp + I2S mic)
#    [*] pwm0   (pin 32: IR LED ring)
#    [*] spi1   (pins 19/21/23/24: XPT2046 touch; skip if the panel is USB)
#  Back -> Save pin changes -> Save and reboot to reconfigure pins
nano$ aplay -l | grep tegrasndt210ref       # after the reboot
```

## 2.6 CPU models and the offline LLM

```bash
nano$ cd ~/beni
nano$ make llama        # llama.cpp llama-server with gcc-9 (-j2, ~40 min) + qwen2.5-0.5b q4_k_m GGUF (~400 MB)
nano$ make models       # as beni: sherpa-onnx KWS/VAD/ASR/TTS/speaker-id, bge-small ONNX, filler clips
nano$ ls /ssd/beni/models
```

## 2.7 Vision engines

Export the detector ONNX files on a PC or on Kaggle, copy them over, then build the engines on the Nano. The full
steps are in [03_MODELS_ENGINES.md](03_MODELS_ENGINES.md):

```bash
nano$ make engines                               # stops vision while building; skips missing ONNX files
nano$ ls /ssd/beni/engines/*.engine
```

## 2.8 ROS 2 Humble container (§7.3)

```bash
nano$ cd ~/beni && make ros-image      # ~2 h the first time, ~5 min afterwards; the image lives in /ssd/docker
nano$ docker images beni-ros
```

## 2.9 Face display

Render the expression clips on the **dev PC**. They need numpy and x264 or ffmpeg. Copy them over, then build on
the Nano:

```bash
pc$   make face-clips                                   # -> jetson/face/assets/clips/*.h264 + *.eyes
pc$   rsync -a jetson/face/assets/clips/ you@beni-jetson:beni/jetson/face/assets/clips/
nano$ cd ~/beni && make face                            # builds beni_face and installs the clips to /ssd/face
nano$ sudo systemctl restart beni-face
```

`beni-face` conflicts with the desktop (`display-manager`). For a headless robot, run
`sudo systemctl set-default multi-user.target`.

## 2.10 Encrypted vault (optional, recommended; §4 item 58)

```bash
nano$ sudo make vault      # LUKS2 file on the SSD, key on the microSD; asks for a recovery passphrase
```

It moves `memory.db` into `/ssd/beni/vault` and points `BENI_DB` in `beni.env` at it. `beni-vault.service` unlocks
it at boot.

## 2.11 Configure and start

```bash
nano$ sudoedit /etc/beni/beni.env
```

| Key | Value |
|---|---|
| `BENI_TOKEN` | a long random string (`openssl rand -hex 24`), the same as the Kaggle secret |
| `BENI_BRAIN_URL` | keep `ws://beni-brain:8765/ws` (the brain's tailnet name) |
| `KAGGLE_KERNEL` | `<kaggle-user>/beni-brain` |
| `HF_TOKEN`, `HF_BACKUP_REPO` | optional: nightly memory backup to a private HF dataset |
| `BENI_DETECTOR` | `yolo26n` (default) or `yolov8n` after the bake-off ([03](03_MODELS_ENGINES.md)) |
| the rest | commented defaults; see the README's "Board features" |

Put the Kaggle API key where the lifecycle manager looks for it:

```bash
nano$ sudo install -d -o beni -g beni -m 700 /home/beni/.kaggle
nano$ sudo install -o beni -g beni -m 600 ~/kaggle.json /home/beni/.kaggle/kaggle.json
nano$ sudo systemctl restart beni-audio beni-vision beni-face beni-ros
nano$ sudo systemctl start beni-agent beni-sched
nano$ make status
```

Next: the brain ([04](04_KAGGLE_BRAIN.md)) and the base firmware ([05](05_FIRMWARE.md)). After those, run
Phase 0 of [ACCEPTANCE.md](../ACCEPTANCE.md).
