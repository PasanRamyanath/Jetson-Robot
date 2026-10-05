#!/usr/bin/env bash
# First-boot tuning for the Jetson Nano 4GB, JetPack 4.6.6 / L4T 32.7.6 (§2.5). Idempotent; reboot afterwards.
# Acceptance: after reboot `free -m` shows >= 3.3 GB available with nothing running.
set -euo pipefail
[ -d /ssd ] || { echo "mount the USB SSD at /ssd first"; exit 1; }
U=${SUDO_USER:-$USER}

# 1. Headless (+0.6-0.8 GB RAM)
sudo systemctl set-default multi-user.target

# 2. Services a robot doesn't need
# bluetooth stays on: phone presence (§4 item 56, BENI_PHONES) pings paired phones; bluetoothd is ~3 MB
for s in cups cups-browsed ModemManager whoopsie apport snapd avahi-daemon; do
  sudo systemctl disable --now "$s" 2>/dev/null || true
done

# 3. UART1 (pins 8/10 -> /dev/ttyTHS1) for the ESP32 base
sudo systemctl disable --now nvgetty 2>/dev/null || true
for g in dialout video i2c gpio audio; do getent group "$g" >/dev/null && sudo usermod -aG "$g" "$U"; done

# 4. MAXN + clocks locked at boot
sudo nvpmodel -m 0
sudo tee /etc/systemd/system/jetson-clocks.service >/dev/null <<'EOF'
[Unit]
Description=Lock Jetson clocks
After=nvpmodel.service
[Service]
Type=oneshot
ExecStart=/usr/bin/jetson_clocks
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable jetson-clocks.service

# 5. Swap on the SSD (keep zram as the first tier, it is cheaper than SSD writes)
if [ ! -f /ssd/swapfile ]; then
  sudo fallocate -l 4G /ssd/swapfile && sudo chmod 600 /ssd/swapfile && sudo mkswap /ssd/swapfile
fi
grep -q '/ssd/swapfile' /etc/fstab || echo '/ssd/swapfile none swap sw,pri=1 0 0' | sudo tee -a /etc/fstab
printf 'vm.swappiness=10\nvm.vfs_cache_pressure=200\nnet.core.rmem_max=4194304\nnet.core.wmem_max=4194304\n' \
  | sudo tee /etc/sysctl.d/99-robot.conf >/dev/null

# 6. RT scheduling for audio
printf '@audio - rtprio 90\n@audio - memlock unlimited\n' | sudo tee /etc/security/limits.d/99-robot.conf >/dev/null

# 7. Base packages (py3.8 venv is created by 05_py38_venv.sh)
sudo apt-get update
sudo apt-get install -y --no-install-recommends python3.8 python3.8-venv python3.8-dev libopus0 libopus-dev \
  gstreamer1.0-plugins-bad gstreamer1.0-plugins-good gstreamer1.0-tools libzmq3-dev libsqlite3-dev \
  python3-pip python3-gi gir1.2-gst-rtsp-server-1.0 alsa-utils i2c-tools smem \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev cmake pkg-config    # beni_face / vision_core builds

# 8. Camera daemon robustness
sudo mkdir -p /etc/systemd/system/nvargus-daemon.service.d
printf '[Service]\nEnvironment=enableCamInfiniteTimeout=1\nRestart=always\nRestartSec=2\n' \
  | sudo tee /etc/systemd/system/nvargus-daemon.service.d/override.conf >/dev/null

# 9. Rootfs on SSD: manual, see github.com/jetsonhacks/rootOnUSB (keep an SD recovery LABEL in extlinux.conf).

# 10. IRQs off the vision/agent cores: pin eth/usb/i2s IRQs to CPU0 at boot (CPUAffinity lives in the units).
sudo tee /etc/systemd/system/beni-irq.service >/dev/null <<'EOF'
[Unit]
Description=Pin device IRQs to CPU0
[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for d in /proc/irq/*/smp_affinity; do echo 1 > $d 2>/dev/null || true; done'
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable beni-irq.service

# 11. 40-pin functions (i2s4, pwm0, spi1): run once, interactive: sudo /opt/nvidia/jetson-io/jetson-io.py

# 12. gcc-9 for llama.cpp (GCC 7.5 is too old)
if ! command -v gcc-9 >/dev/null; then
  sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test && sudo apt-get update && sudo apt-get install -y gcc-9 g++-9
fi

# 13. Hardware watchdog
sudo sed -i 's/^#\?RuntimeWatchdogSec=.*/RuntimeWatchdogSec=30/' /etc/systemd/system.conf

# 14. Wi-Fi power save off (§15.2 #9): ~0.2 W, but no 100-300 ms latency spikes. NM re-applies it on every connect.
if [ -d /etc/NetworkManager/conf.d ]; then
  printf '[connection]\nwifi.powersave = 2\n' | sudo tee /etc/NetworkManager/conf.d/99-beni-wifi.conf >/dev/null
fi
sudo iw dev wlan0 set power_save off 2>/dev/null || true

# Data dirs
# Owned by the service user once `make jetson-install` has created it (the services write here), else by you.
O=$(id -u beni >/dev/null 2>&1 && echo beni || echo "$U")
sudo mkdir -p /ssd/beni/{models,engines,rec,thumbs,backup,logs} /ssd/maps && sudo chown -R "$O" /ssd/beni /ssd/maps
sudo systemctl daemon-reload
echo "Done. Reboot now."
