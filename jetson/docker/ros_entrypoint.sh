#!/bin/bash
# §7.3: underlay (ROS) -> deps -> beni. umask 0 so the agent (non-root) can connect to the ipc sockets in /tmp/beni.
set -e
umask 0000
source /opt/beni/setup.bash
mkdir -p /tmp/beni /maps
exec "$@"
