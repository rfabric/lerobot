# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

.PHONY: tests

PYTHON_PATH := $(shell which python)

# If uv is installed and a virtual environment exists, use it
# UV_CHECK := $(shell command -v uv)
# ifneq ($(UV_CHECK),)
# 	PYTHON_PATH := $(shell .venv/bin/python)
# endif


# If uv is installed and a virtual environment exists, use it.
# Never run `python` inside $(shell …) — an interactive interpreter
# steals the TTY and looks like `make` "turned into" Python (>>>).
UV_CHECK := $(shell command -v uv 2>/dev/null)
ifneq ($(UV_CHECK),)
ifneq ($(wildcard .venv/bin/python),)
	PYTHON_PATH := $(CURDIR)/.venv/bin/python
endif
endif

export PATH := $(dir $(PYTHON_PATH)):$(PATH)

DEVICE ?= cpu

LEROBOT_DIR ?= $(CURDIR)
LEROBOT_PYTHON ?= $(LEROBOT_DIR)/.venv/bin/python
LEROBOT_SOCKET ?= /tmp/rfabric/control.sock
LEROBOT_LEFT_PORT ?= /dev/tty.wchusbserial5AAF2202361
LEROBOT_RIGHT_PORT ?= /dev/tty.wchusbserial5AE70450321
# Vendored in the lerobot fork under assets/SO-ARM100 (not assets/Simulation).
LEROBOT_URDF ?= $(LEROBOT_DIR)/assets/SO-ARM100/Simulation/SO101/so101_new_calib.urdf
# Passed to ``--robot-id`` so left/right followers get separate calibration JSON
# (``<id>_left.json`` / ``<id>_right.json``). Never leave this empty.
LEROBOT_ROBOT_ID ?= rfabric_bimanual
ARGS ?=




build-user:
	docker build -f docker/Dockerfile.user -t lerobot-user .

build-internal:
	docker build -f docker/Dockerfile.internal -t lerobot-internal .

test-end-to-end:
	${MAKE} DEVICE=$(DEVICE) test-act-ete-train
	${MAKE} DEVICE=$(DEVICE) test-act-ete-train-resume
	${MAKE} DEVICE=$(DEVICE) test-act-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-diffusion-ete-train
	${MAKE} DEVICE=$(DEVICE) test-diffusion-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-tdmpc-ete-train
	${MAKE} DEVICE=$(DEVICE) test-tdmpc-ete-eval
	${MAKE} DEVICE=$(DEVICE) test-smolvla-ete-train
	${MAKE} DEVICE=$(DEVICE) test-smolvla-ete-eval

test-act-ete-train:
	lerobot-train \
		--policy.type=act \
		--policy.dim_model=64 \
		--policy.n_action_steps=20 \
		--policy.chunk_size=20 \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=aloha \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=4 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_freq=2 \
		--save_checkpoint=true \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/act/

test-act-ete-train-resume:
	lerobot-train \
		--config_path=tests/outputs/act/checkpoints/000002/pretrained_model/train_config.json \
		--resume=true

test-act-ete-eval:
	lerobot-eval \
		--policy.path=tests/outputs/act/checkpoints/000004/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=aloha \
		--env.episode_length=5 \
		--eval.n_episodes=1 \
		--eval.batch_size=1

test-diffusion-ete-train:
	lerobot-train \
		--policy.type=diffusion \
		--policy.down_dims='[64,128,256]' \
		--policy.diffusion_step_embed_dim=32 \
		--policy.num_inference_steps=10 \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=pusht \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/pusht \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=2 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_checkpoint=true \
		--save_freq=2 \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/diffusion/

test-diffusion-ete-eval:
	lerobot-eval \
		--policy.path=tests/outputs/diffusion/checkpoints/000002/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=pusht \
		--env.episode_length=5 \
		--eval.n_episodes=1 \
		--eval.batch_size=1

test-tdmpc-ete-train:
	lerobot-train \
		--policy.type=tdmpc \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=pusht \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/pusht_image \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=2 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_checkpoint=true \
		--save_freq=2 \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/tdmpc/

test-tdmpc-ete-eval:
	lerobot-eval \
		--policy.path=tests/outputs/tdmpc/checkpoints/000002/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=pusht \
		--env.episode_length=5 \
		--env.observation_height=96 \
        --env.observation_width=96 \
		--eval.n_episodes=1 \
		--eval.batch_size=1


test-smolvla-ete-train:
	lerobot-train \
		--policy.type=smolvla \
		--policy.n_action_steps=20 \
		--policy.chunk_size=20 \
		--policy.device=$(DEVICE) \
		--policy.push_to_hub=false \
		--env.type=aloha \
		--env.episode_length=5 \
		--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
		--dataset.image_transforms.enable=true \
		--dataset.episodes="[0]" \
		--batch_size=2 \
		--steps=4 \
		--eval_freq=2 \
		--eval.n_episodes=1 \
		--eval.batch_size=1 \
		--save_freq=2 \
		--save_checkpoint=true \
		--log_freq=1 \
		--wandb.enable=false \
		--output_dir=tests/outputs/smolvla/

test-smolvla-ete-eval:
	lerobot-eval \
		--policy.path=tests/outputs/smolvla/checkpoints/000004/pretrained_model \
		--policy.device=$(DEVICE) \
		--env.type=aloha \
		--env.episode_length=5 \
		--eval.n_episodes=1 \
		--eval.batch_size=1



LEADER_LEFT_PORT=/dev/tty.wchusbserial5AAF2181801
FOLLOWER_LEFT_PORT=/dev/tty.wchusbserial5AAF2202361

LEADER_RIGHT_PORT=/dev/tty.wchusbserial5AE70493031
FOLLOWER_RIGHT_PORT=/dev/tty.wchusbserial5AE70450321

port:
	lerobot-find-port

setup\:leader\:left:
	lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=$(LEADER_LEFT_PORT)

setup\:leader\:right:
	lerobot-setup-motors \
    --teleop.type=so101_leader \
    --teleop.port=$(LEADER_RIGHT_PORT)


setup\:follower\:left:
	lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=$(FOLLOWER_LEFT_PORT)

setup\:follower\:right:
	lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=$(FOLLOWER_RIGHT_PORT)


calibrate\:leader\:left:
	lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=$(LEADER_LEFT_PORT) \
    --teleop.id=left_arm_lead

calibrate\:leader\:right:
	lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=$(LEADER_RIGHT_PORT) \
    --teleop.id=right_arm_lead

calibrate\:follower\:left:
	lerobot-calibrate \
		--robot.type=so101_follower \
		--robot.port=$(FOLLOWER_LEFT_PORT) \
		--robot.id=left_arm

calibrate\:follower\:right:
	lerobot-calibrate \
		--robot.type=so101_follower \
		--robot.port=$(FOLLOWER_RIGHT_PORT) \
		--robot.id=right_arm

calibrate\:rfabric\:left:
	lerobot-calibrate \
		--robot.type=so101_follower \
		--robot.port=$(FOLLOWER_LEFT_PORT) \
		--robot.id=rfabric_bimanual_left

calibrate\:rfabric\:right:
	lerobot-calibrate \
		--robot.type=so101_follower \
		--robot.port=$(FOLLOWER_RIGHT_PORT) \
		--robot.id=rfabric_bimanual_right

calibrate\:rfabric: calibrate\:rfabric\:left calibrate\:rfabric\:right

tele\:left:
	lerobot-teleoperate \
		--robot.type=so101_follower \
		--robot.port=$(FOLLOWER_LEFT_PORT) \
		--robot.id=left_arm \
		--teleop.type=so101_leader \
		--teleop.port=$(LEADER_LEFT_PORT) \
		--teleop.id=left_arm_lead \

# --display_data=true

tele\:right:
	lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=$(FOLLOWER_RIGHT_PORT) \
    --robot.id=right_arm \
    --teleop.type=so101_leader \
    --teleop.port=$(LEADER_RIGHT_PORT) \
    --teleop.id=right_arm_lead \

# --display_data=true
tele\:bi:
	lerobot-teleoperate \
		--robot.type=bi_so_follower \
		--robot.left_arm_config.port=$(FOLLOWER_LEFT_PORT) \
		--robot.right_arm_config.port=$(FOLLOWER_RIGHT_PORT) \
		--robot.id=bimanual_follower \
		--teleop.type=bi_so_leader \
		--teleop.left_arm_config.port=$(LEADER_LEFT_PORT) \
		--teleop.right_arm_config.port=$(LEADER_RIGHT_PORT) \
		--teleop.id=bimanual_leader \

# --display_data=true


# --robot.cameras='{
#   left: {"type": "opencv", "index_or_path": 0, "width": 1920, "height": 1080, "fps": 30},
#   top: {"type": "opencv", "index_or_path": 1, "width": 1920, "height": 1080, "fps": 30},
#   right: {"type": "opencv", "index_or_path": 2, "width": 1920, "height": 1080, "fps": 30}
# }' \


tele\:cam:
	lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.wchusbserial5AAF2202361 \
    --robot.id=follower \
		--robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.wchusbserial5AAF2181801 \
    --teleop.id=leader \
		--display_data=true


tele\:bi\:cam:
	lerobot-teleoperate \
    --robot.type=bi_so_follower \
		--robot.left_arm_config.port=$(FOLLOWER_LEFT_PORT) \
		--robot.right_arm_config.port=$(FOLLOWER_RIGHT_PORT) \
		--robot.id=bimanual_follower \
		--robot.cameras="{ left: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, right: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
		--teleop.type=bi_so_leader \
		--teleop.left_arm_config.port=$(LEADER_LEFT_PORT) \
		--teleop.right_arm_config.port=$(LEADER_RIGHT_PORT) \
		--teleop.id=bimanual_leader \
		--display_data=true

DATASET_ROOT := data/$(shell date +%Y%m%d_%H%M%S)

record:
	lerobot-record \
    --robot.type=bi_so_follower \
    --robot.left_arm_config.port=$(FOLLOWER_LEFT_PORT) \
		--robot.right_arm_config.port=$(FOLLOWER_RIGHT_PORT) \
    --robot.id=bimanual_follower \
    --robot.left_arm_config.cameras="{ left: {type: opencv, index_or_path: 2, width: 1920, height: 1080, fps: 30, rotation: 180}, top: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30, rotation: 0}}" \
    --robot.right_arm_config.cameras="{ right: {type: opencv, index_or_path: 1, width: 1920, height: 1080, fps: 30, rotation: 180}}" \
    --teleop.type=bi_so_leader \
    --teleop.right_arm_config.port=$(LEADER_RIGHT_PORT) \
		--teleop.left_arm_config.port=$(LEADER_LEFT_PORT) \
    --teleop.id=bimanual_leader \
		--dataset.repo_id=lerobot/bimanual \
		--dataset.root=$(DATASET_ROOT) \
    --dataset.num_episodes=1 \
		--dataset.reset_time_s=5 \
    --dataset.single_task="Bimanual task" \
		--dataset.push_to_hub=false

# --dataset.episode_time_s=45 \
# --dataset.remote.type=s3 \
# --dataset.remote.s3.bucket=lute-datasets-raw-ingest-dev

# --dataset.num_image_writer_processes=4 \
# --display_data=true \

replay:
	lerobot-replay \
    --robot.type=bi_so_follower \
    --robot.left_arm_config.port=$(FOLLOWER_LEFT_PORT) \
		--robot.right_arm_config.port=$(FOLLOWER_RIGHT_PORT) \
    --robot.id=bimanual_follower \
		--dataset.repo_id=lerobot/bimanual \
		--dataset.root=data/20260403_172709 \
		--dataset.episode=0

viz:
	lerobot-dataset-viz \
    --repo-id lerobot/test2 \
    --mode local \
    --episode-index 0


run-arms: ## Start the local lerobot sidecar bound to the agent's UDS.
	@mkdir -p $(dir $(LEROBOT_SOCKET))
	cd $(LEROBOT_DIR) && $(LEROBOT_PYTHON) -m lerobot.teleoperators.rfabric_remote.bimanual_so101 \
		--left-port=$(LEROBOT_LEFT_PORT) \
		--right-port=$(LEROBOT_RIGHT_PORT) \
		--urdf=$(LEROBOT_URDF) \
		--socket-path=$(LEROBOT_SOCKET) \
		--robot-id=$(LEROBOT_ROBOT_ID) \
		$(ARGS)
