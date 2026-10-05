#!/usr/bin/env bash
# §7.1 budget table: CPU % (of one core) and RSS per Beni service over SECS (default 30), from /proc, whole cgroup.
source "$(dirname "$0")/common.sh"
SECS=${1:-30}
HZ=$(getconf CLK_TCK)
declare -A budget=([beni-audio]="12-20 %, 25 MB" [beni-face]="3-6 %, 40-60 MB" [beni-vision]="40-60 %, 150-500 MB"
                   [beni-agent]="50-100 %, 350-550 MB" [beni-sched]="1 %, 25 MB" [beni-ros]="80-140 %, 450-700 MB"
                   [beni-llm]="idle 0 %, 450-550 MB" [mediamtx]="2-10 %, 60 MB")
pids_of() {   # all pids in the unit's cgroup (docker containers: the container's cgroup); none if it isn't running
  local u=$1 cg
  if [ "$u" = beni-ros ]; then
    cg=$(docker inspect -f '{{.Id}}' beni-ros 2>/dev/null) && cat /sys/fs/cgroup/cpu*/docker/"$cg"/cgroup.procs 2>/dev/null
  else
    cg=$(systemctl show -p ControlGroup --value "$u.service" 2>/dev/null)
    [ -n "$cg" ] && cat /sys/fs/cgroup/systemd"$cg"/cgroup.procs 2>/dev/null
  fi
  return 0      # set -e + pipefail: a stopped unit must give the "not running" row, not abort the run
}
ticks() { local s=0 p; for p in "$@"; do s=$(( s + $(awk '{print $14 + $15}' /proc/$p/stat 2>/dev/null || echo 0) )); done; echo $s; }
rss() { local s=0 p; for p in "$@"; do s=$(( s + $(awk '/VmRSS/{print $2}' /proc/$p/status 2>/dev/null || echo 0) )); done; echo $(( s / 1024 )); }
declare -A t0
for u in "${!budget[@]}"; do t0[$u]=$(ticks $(pids_of "$u")); done
sleep "$SECS"
for u in $(printf '%s\n' "${!budget[@]}" | sort); do
  p=$(pids_of "$u" | tr '\n' ' ')
  [ -n "$p" ] || { row "$u" "CPU %, RSS MB" "not running" "${budget[$u]}"; continue; }
  cpu=$(( ( $(ticks $p) - ${t0[$u]} ) * 100 / (HZ * SECS) ))
  row "$u" "CPU % of a core, RSS MB (${SECS}s)" "$cpu %, $(rss $p) MB" "${budget[$u]}"
done
