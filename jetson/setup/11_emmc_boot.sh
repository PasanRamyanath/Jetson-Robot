#!/usr/bin/env bash
# eMMC Nano (Waveshare kit, P3448-0002) with the rootfs on a microSD card or SSD (§2.5 step 9). Depending on how the
# rootfs was moved, U-Boot may still read the kernel, DTB and extlinux.conf from the eMMC's /boot. Then jetson-io's
# pin changes (i2s4/pwm0/spi1) and kernel updates land in the running rootfs's /boot and silently do nothing.
#   sudo bash 11_emmc_boot.sh check          tags this /boot; reboot, run check again: says whether you need sync
#   sudo CONFIRM=1 bash 11_emmc_boot.sh sync   copy /boot to the eMMC, keep root= on the running rootfs
# Run sync after every jetson-io change and every nvidia-l4t-kernel* update. EMMC=/dev/mmcblk0p1 by default.
set -euo pipefail
[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
EMMC=${EMMC:-/dev/mmcblk0p1}
EXT=/boot/extlinux/extlinux.conf
TAG=beni_bootsrc=rootfs
ROOTDEV=$(findmnt -no SOURCE /)
[ "$ROOTDEV" != "$EMMC" ] || { echo "the rootfs is on $EMMC itself: nothing to do"; exit 0; }
[ -b "$EMMC" ] || { echo "$EMMC not found (lsblk; set EMMC=...)"; exit 1; }

case "${1:-}" in
check)
  if grep -qw "$TAG" /proc/cmdline; then
    echo "OK: this rootfs's own /boot is what boots ($ROOTDEV). No sync needed."
  elif grep -qw "$TAG" "$EXT"; then
    echo "The eMMC's /boot is what boots. Run: sudo CONFIRM=1 bash $0 sync  (after jetson-io and kernel updates)"
  else
    cp -n "$EXT" "$EXT.beni-orig"
    sed -i "s/^\([[:space:]]*APPEND\) /\1 $TAG /" "$EXT"
    echo "tagged $EXT; now: sudo reboot, then run: sudo bash $0 check"
  fi
  ;;
sync)
  [ "${CONFIRM:-0}" = 1 ] || { echo "rewrites $EMMC:/boot (a backup is kept); re-run with CONFIRM=1"; exit 1; }
  MNT=/mnt/beni-emmc
  mkdir -p "$MNT" && mount "$EMMC" "$MNT"
  trap 'umount "$MNT"' EXIT
  E="$MNT$EXT"
  [ -f "$E" ] || { echo "no $EXT on $EMMC"; exit 1; }
  ROOTARG=$(grep -o "root=[^ ]*" /proc/cmdline | tail -1)     # the last root= wins; it boots this rootfs today
  cp "$E" "$E.beni-$(date +%Y%m%d-%H%M%S)"
  rsync -a --exclude=extlinux/ /boot/ "$MNT/boot/"
  python3 - "$EXT" "$E" "$ROOTARG" "$EMMC" "$TAG" <<'PY'
import re, sys
src, dst, rootarg, emmc, tag = sys.argv[1:]
s = open(src).read().replace(" " + tag, "")
s = re.sub(r"root=\S+", rootarg, s)
m = re.search(r"(?ms)^LABEL \S+\n.*?(?=^LABEL |\Z)", s)
if m and "LABEL emmc-rootfs" not in s:            # serial-console fallback to the eMMC's own rootfs
    rec = re.sub(r"^LABEL \S+", "LABEL emmc-rootfs", m.group(0), count=1)
    rec = re.sub(r"(?m)^(\s*MENU LABEL ).*", r"\1eMMC rootfs (recovery)", rec)
    s = s.rstrip("\n") + "\n\n" + re.sub(r"root=\S+", "root=" + emmc, rec)
open(dst, "w").write(s)
PY
  grep -nE "^(DEFAULT|LABEL)|FDT|root=" "$E"
  echo "synced /boot -> $EMMC. Reboot."
  ;;
*) echo "usage: $0 check | sync"; exit 1 ;;
esac
