#!/usr/bin/env bash
# Docker data-root on the SSD + NVIDIA runtime as default (ROS2 Humble container, §7.3).
set -euo pipefail
sudo mkdir -p /ssd/docker
sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{
  "data-root": "/ssd/docker",
  "default-runtime": "nvidia",
  "runtimes": {"nvidia": {"path": "nvidia-container-runtime", "runtimeArgs": []}},
  "log-driver": "local",
  "log-opts": {"max-size": "10m", "max-file": "3"}
}
JSON
sudo usermod -aG docker "${SUDO_USER:-$USER}"
sudo systemctl restart docker
# Base image built for r32.7.1 runs on 32.7.6 (same userspace ABI); building ROS from source on the Nano fails.
docker pull dustynv/ros:humble-ros-base-l4t-r32.7.1
