#!/usr/bin/env bash
# §3.17 SE row, §4 item 58, §12.3: encrypted vault for biometrics and memory (memory.db holds the face/voice
# exemplars, the people and every fact; backups too). A LUKS2 *file* container on the SSD (aes-xts-plain64; the A57
# ARMv8 crypto extensions and tegra-se take the load), so the rootOnUSB layout needs no repartitioning.
# The key file lives on the microSD boot card, not the SSD: a stolen SSD alone is unreadable. A passphrase slot is
# added as the recovery path.
#   sudo bash 09_vault.sh create     once: container + key + passphrase, move memory.db in, point beni.env at it
#   sudo bash 09_vault.sh open|close|status   (beni-vault.service runs open/close at boot/shutdown)
set -euo pipefail
IMG=${IMG:-/ssd/beni-vault.img}
SIZE_GB=${SIZE_GB:-8}
MNT=${MNT:-/ssd/beni/vault}
NAME=beni_vault
SD_DEV=${SD_DEV:-/dev/mmcblk0p1}
SD_MNT=/run/beni-sd
KEY_REL=beni/vault.key
ENV=/etc/beni/beni.env

sd_mount() {  # rw only while creating the key; ro otherwise
  mkdir -p "$SD_MNT"
  mountpoint -q "$SD_MNT" || mount -o "${1:-ro}" "$SD_DEV" "$SD_MNT"
}
sd_umount() { mountpoint -q "$SD_MNT" && umount "$SD_MNT" || true; }

open_vault() {
  [ -e "/dev/mapper/$NAME" ] || {
    sd_mount ro
    cryptsetup open --type luks2 --key-file "$SD_MNT/$KEY_REL" --allow-discards "$IMG" "$NAME"
    sd_umount
  }
  mkdir -p "$MNT"
  mountpoint -q "$MNT" || mount -o noatime,commit=30 "/dev/mapper/$NAME" "$MNT"
}

close_vault() {
  mountpoint -q "$MNT" && umount "$MNT"
  [ -e "/dev/mapper/$NAME" ] && cryptsetup close "$NAME"
  true
}

set_env() {  # key value
  grep -q "^$1=" "$ENV" && sed -i "s|^$1=.*|$1=$2|" "$ENV" || echo "$1=$2" >> "$ENV"
}

create() {
  [ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
  [ -e "$IMG" ] && { echo "$IMG exists"; exit 1; }
  grep -qw aes /proc/cpuinfo || echo "note: no ARMv8 AES in /proc/cpuinfo; expect slower I/O"
  systemctl stop beni-agent beni-sched 2>/dev/null || true
  sd_mount rw
  mkdir -p "$SD_MNT/beni"
  [ -s "$SD_MNT/$KEY_REL" ] || { head -c 64 /dev/urandom > "$SD_MNT/$KEY_REL"; chmod 400 "$SD_MNT/$KEY_REL"; }
  fallocate -l "${SIZE_GB}G" "$IMG"
  chmod 600 "$IMG"
  cryptsetup luksFormat --batch-mode --type luks2 --cipher aes-xts-plain64 --key-size 512 --pbkdf argon2i \
    --pbkdf-memory 65536 --key-file "$SD_MNT/$KEY_REL" "$IMG"
  echo "Recovery passphrase (keep it off the robot):"
  cryptsetup luksAddKey --key-file "$SD_MNT/$KEY_REL" "$IMG"
  cryptsetup open --type luks2 --key-file "$SD_MNT/$KEY_REL" --allow-discards "$IMG" "$NAME"
  sd_umount
  mkfs.ext4 -q -L beni-vault -m 0 "/dev/mapper/$NAME"
  open_vault
  for f in /ssd/beni/memory.db /ssd/beni/memory.db-wal /ssd/beni/memory.db-shm; do
    [ -e "$f" ] && mv "$f" "$MNT/"
  done
  [ -d /ssd/beni/backup ] && mv /ssd/beni/backup "$MNT/backup"
  mkdir -p "$MNT/backup"
  chown -R beni:beni "$MNT"
  set_env BENI_DB "$MNT/memory.db"
  set_env BENI_BACKUP_DIR "$MNT/backup"
  install -m 644 "$(dirname "$0")/../systemd/beni-vault.service" /etc/systemd/system/
  systemctl daemon-reload && systemctl enable beni-vault
  echo "vault ready at $MNT; BENI_DB and BENI_BACKUP_DIR updated in $ENV"
}

case "${1:-status}" in
  create) create ;;
  open) open_vault ;;
  close) close_vault ;;
  status) cryptsetup status "$NAME" || true; mountpoint "$MNT" || true ;;
  *) echo "usage: $0 create|open|close|status"; exit 2 ;;
esac
