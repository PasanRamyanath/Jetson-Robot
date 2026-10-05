# Beni robot: one entry point for dev (any PC), the Jetson Nano (JetPack 4.6.x) and the Kaggle brain.
#   make test | lint                        dev machine, Python >= 3.10 (stub models, no GPU)
#   sudo make jetson-install                on the Nano: /opt/beni symlink, user, units, tmpfiles, sudoers, env
#   make ssd-root PART=/dev/sda1 CONFIRM=1  on the Nano, once, then reboot
#   make jetson-setup | models | engines    on the Nano, once (in this order)
#   make emmc-boot-check | emmc-boot-sync   eMMC Nano with the rootfs on SD/SSD: after jetson-io and kernel updates
#   make face                               on the Nano: build beni_face, install clips to /ssd/face
#   make vision-core                        on the Nano (Phase 3): build the MMAPI/TensorRT vision binary
#   make vault                              on the Nano (Phase 5): LUKS2 vault for memory.db + biometrics
#   make ros-image                          on the Nano: ROS 2 Humble container (~2 h first time, then ~5 min)
#   make wheelhouse-push | kaggle-push KAGGLE_KERNEL=you/beni-brain   upload the wheelhouse; push the brain by hand
SHELL := /bin/bash
ROOT  := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
PY    ?= python3
UNITS := beni-agent beni-sched beni-audio beni-vision beni-vision-core beni-face beni-ros beni-llm beni-vault
KERNEL_DIR := $(ROOT)/kaggle

.PHONY: test lint e2e jetson-install jetson-setup models engines llama face face-clips vault vision-core bench emmc-boot-check emmc-boot-sync ros-image wheelhouse wheelhouse-push kaggle-push brain-stub status ssd-root

test:
	$(PY) -m pytest -q

e2e:
	$(PY) -m pytest -q tests/brain/test_gateway_e2e.py

lint:
	$(PY) -m pyflakes shared kaggle jetson/agent jetson/tools jetson/vision jetson/engines tests

brain-stub:                     ## local brain with stub models on :8765 (point BENI_BRAIN_URL here to test the agent)
	PYTHONPATH=$(ROOT)/shared:$(ROOT)/kaggle BENI_RUN_DIR=/tmp/beni-brain BENI_DB=/tmp/beni-brain/memory.db \
	  $(PY) -m beni_brain.gateway --stub --host 127.0.0.1

# ---------------------------------------------------------------- Jetson
jetson-install:
	@[ "$$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
	id beni >/dev/null 2>&1 || useradd -m -G audio,video,dialout,i2c,gpio -s /bin/bash beni || useradd -m -G audio,video,dialout -s /bin/bash beni
	ln -sfn $(ROOT) /opt/beni
	mkdir -p /ssd/beni/models /ssd/beni/engines /ssd/beni/logs /ssd/maps /ssd/face /etc/beni
	chown -R beni:beni /ssd/beni /ssd/maps /ssd/face
	[ -f /etc/beni/beni.env ] || install -m 600 -o beni -g beni jetson/systemd/beni.env.example /etc/beni/beni.env
	install -m 644 jetson/systemd/beni-tmpfiles.conf /etc/tmpfiles.d/beni.conf && systemd-tmpfiles --create /etc/tmpfiles.d/beni.conf
	install -m 440 jetson/systemd/beni-sudoers /etc/sudoers.d/beni && visudo -cf /etc/sudoers.d/beni
	for u in $(UNITS); do install -m 644 jetson/systemd/$$u.service /etc/systemd/system/; done
	systemctl daemon-reload
	systemctl enable beni-audio beni-vision beni-face beni-ros beni-agent beni-sched    # beni-llm is on demand
	@echo "edit /etc/beni/beni.env, then: sudo systemctl start beni-agent"

ssd-root:                       ## once, before jetson-setup: make ssd-root PART=/dev/sda1, then reboot
	sudo CONFIRM=$(CONFIRM) bash jetson/setup/02_root_on_ssd.sh $(PART)

jetson-setup:
	sudo bash jetson/setup/00_jetpack.sh        # CUDA/TensorRT/MMAPI/DeepStream/pyds if missing (eMMC flashes)
	sudo bash jetson/setup/01_system_tune.sh
	sudo bash jetson/setup/03_docker.sh
	sudo TS_AUTHKEY="$$TS_AUTHKEY" bash jetson/setup/04_tailscale.sh    # sudo drops the env otherwise
	bash jetson/setup/05_py38_venv.sh
	bash jetson/setup/08_mediamtx.sh
	bash jetson/setup/10_deepstream_yolo.sh

models:
	sudo -u beni env BENI_MODELS=/ssd/beni/models bash jetson/setup/07_models.sh

engines:
	sudo -u beni bash jetson/engines/build_all.sh

face:                           ## §3.11; clips come from `make face-clips` (dev PC) or jetson/face/assets/clips
	mkdir -p jetson/face/build && cd jetson/face/build && cmake -DCMAKE_BUILD_TYPE=Release .. && $(MAKE) -j2   # cmake 3.10: no -S/-B
	sudo install -m 644 -o beni -g beni jetson/face/assets/clips/*.h264 jetson/face/assets/clips/*.eyes /ssd/face/

vision-core:                    ## §5.5 Phase 3: MMAPI + TensorRT vision (replaces beni-vision; see README)
	mkdir -p jetson/vision_core/build && cd jetson/vision_core/build && cmake -DCMAKE_BUILD_TYPE=Release .. && $(MAKE) -j3

vault:                          ## §4 item 58: encrypted memory/biometrics vault, key on the microSD card
	sudo bash jetson/setup/09_vault.sh create

face-clips:                     ## dev PC: numpy + x264 (or ffmpeg/libx264)
	$(PY) jetson/face/assets/make_clips.py

ros-image:                      ## §7.3; docker data-root must be on the SSD
	DOCKER_BUILDKIT=1 docker build -f jetson/docker/Dockerfile.ros --build-arg JOBS=$${JOBS:-2} -t beni-ros:latest $(ROOT)

llama:
	bash jetson/llm/build_llama.sh

bench:                          ## §14.3 on the Nano; MODE=active|idle labels the tegrastats run
	bash jetson/tools/bench/run_all.sh

emmc-boot-check:
	sudo bash jetson/setup/11_emmc_boot.sh check

emmc-boot-sync:
	sudo CONFIRM=1 bash jetson/setup/11_emmc_boot.sh sync

status:
	systemctl --no-pager status $(UNITS) | grep -E "●|Active:"

# ---------------------------------------------------------------- Kaggle
wheelhouse:
	bash kaggle/wheelhouse/build_wheelhouse.sh

wheelhouse-push:
	bash kaggle/wheelhouse/build_wheelhouse.sh --push

kaggle-push:                    ## by hand, KAGGLE_KERNEL=you/beni-brain; normally the lifecycle manager (same rewrite)
	@[ -n "$(KAGGLE_KERNEL)" ] || { echo "set KAGGLE_KERNEL=<user>/beni-brain"; exit 1; }
	rm -rf /tmp/beni-kernel && mkdir -p /tmp/beni-kernel && cp $(KERNEL_DIR)/brain_notebook.py /tmp/beni-kernel/
	sed -e 's#"id": "[^"]*"#"id": "$(KAGGLE_KERNEL)"#' \
	    -e 's#YOUR_KAGGLE_USERNAME#$(firstword $(subst /, ,$(KAGGLE_KERNEL)))#g' \
	    $(KERNEL_DIR)/kernel-metadata.json > /tmp/beni-kernel/kernel-metadata.json
	kaggle kernels push -p /tmp/beni-kernel
