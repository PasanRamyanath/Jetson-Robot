#!/usr/bin/env bash
# Rootfs on the USB SSD, boot stays on microSD (§2.5 step 9, same approach as jetsonhacks/rootOnUSB).
#   sudo CONFIRM=1 jetson/setup/02_root_on_ssd.sh /dev/sda1
# The partition must already be ext4 (this script never formats). Copies the running rootfs, points the default
# extlinux entry at the SSD by PARTUUID and keeps the SD rootfs as a "recovery" boot entry. Reboot afterwards,
# then run 01_system_tune.sh (/ssd becomes a plain directory on the SSD root).
# eMMC Nano (Waveshare) already running from SD: mount the SSD at /ssd instead (instructions.md C3), or run
# 11_emmc_boot.sh check first: if the eMMC's /boot is what boots, this edit needs 11_emmc_boot.sh sync after it.
set -euo pipefail
PART=${1:?usage: $0 /dev/sdXN}
[ "$(id -u)" = 0 ] || { echo "run as root"; exit 1; }
[ "$(blkid -o value -s TYPE "$PART")" = ext4 ] || { echo "$PART is not ext4 (mkfs.ext4 it yourself first)"; exit 1; }
[ "$(findmnt -no SOURCE /)" != "$PART" ] || { echo "already running from $PART"; exit 0; }
[ "${CONFIRM:-0}" = 1 ] || { echo "will overwrite files on $PART; re-run with CONFIRM=1"; exit 1; }

MNT=/mnt/beni-ssd-root
mkdir -p "$MNT"
mount "$PART" "$MNT"
trap 'umount "$MNT"' EXIT
rsync -axHAWX --numeric-ids --info=progress2 --exclude=/proc/* --exclude=/sys/* --exclude=/dev/* \
  --exclude=/run/* --exclude=/tmp/* --exclude=/mnt/* --exclude=/media/* --exclude=/lost+found / "$MNT"
mkdir -p "$MNT/ssd"
sed -i '\#[[:space:]]/ssd[[:space:]]#d' "$MNT/etc/fstab"          # /ssd is now just a directory on the SSD root

UUID=$(blkid -o value -s PARTUUID "$PART")
EXT=/boot/extlinux/extlinux.conf
cp -n "$EXT" "$EXT.sd-orig"
if ! grep -q "LABEL ssd" "$EXT"; then
  python3 - "$EXT" "$UUID" <<'PY'
import re, sys
path, uuid = sys.argv[1], sys.argv[2]
s = open(path).read()
m = re.search(r"(?ms)^LABEL primary\n.*?(?=^LABEL |\Z)", s)
primary = m.group(0)
ssd = re.sub(r"root=\S+", "root=PARTUUID=" + uuid, primary.replace("LABEL primary", "LABEL ssd", 1))
ssd = ssd.replace("MENU LABEL primary kernel", "MENU LABEL SSD rootfs")
recovery = primary.replace("LABEL primary", "LABEL recovery", 1)
s = s.replace(primary, ssd + "\n" + recovery)
s = re.sub(r"(?m)^DEFAULT \S+", "DEFAULT ssd", s)
open(path, "w").write(s)
PY
fi
grep -nE "^(DEFAULT|LABEL)|root=" "$EXT"
echo "done: reboot; if the SSD fails to boot, pick 'recovery' on the serial console."
