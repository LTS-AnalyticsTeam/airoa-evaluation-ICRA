# Deploy Guide (Real HSR)

Real HSR deployment guide.

## 0. Web GUI Quick Start (Recommended)

Minimum requirements:

- `python3` is available
- `deploy/models.json` and `deploy/evaluation_tasks.json` are already edited for your environment

Initial setup example:

```bash
cp deploy/models.example.json deploy/models.json
cp deploy/evaluation_tasks.example.json deploy/evaluation_tasks.json
```

Start:

```bash
python3 deploy/eval_competition_web.py \
  --models deploy/models.json \
  --tasks deploy/evaluation_tasks.json \
  --runs-per-sht 2 \
  --hsr-ip 100.119.167.94
```

Options used in this quick-start command:

- `--models`: model list JSON path.
- `--tasks`: SHT/PA task definition JSON path.
- `--runs-per-sht 2`: run each SHT twice (`default: 1`).
- `--hsr-ip 100.119.167.94`: sets robot IP and auto-derives `ROS_MASTER_URI=http://100.119.167.94:11311`.

Access:

- Open `http://<host-machine-ip>:8080` from any device on the same network

Useful optional flags for web run:

- `--host 0.0.0.0 --port 8080`: bind address/port
- `--max-running N`: maximum concurrently running model server containers
- `--start-port P`: base port for model servers
- `--skip-client-up`: reuse already running `airoa_hsr_client`
- `--disable-gpu`: run model servers without `--gpus all`
- `--ros-ip-choice 2`: select the second detected host IP candidate

GUI flow:

1. Select a model in `Select Model`.
2. Click `Start evaluating <model>` (use `Choose another model` if you need to reselect).
3. For the first PA after model load, the timer starts after first action detection. After that, each PA timer starts when the PA starts.
4. For each PA, record `Success` or `Fail`.
5. If you entered a wrong PA result, enable `Edit Enabled` in `PA Result Records`, then switch the row radio button (`Success`/`Fail`). The change is saved immediately.
6. If you closed SHT review with `Close (Edit Later)`, you can still fix PA records from `PA Result Records` before continuing.
7. `SHT Fail (Skip Rest)` is shown only on the `Next PA` screen after a PA is marked `Fail`.
8. At SHT end, choose `SHT Success` or `SHT Fail`, then confirm with `Confirm and Next SHT`.
9. After each SHT finishes all eval repeats, complete `Reset Robot`, review the SHT summary (`sht_success_rate` + per-PA success rates), then continue.
10. After the final SHT summary of a model, the GUI returns to model selection.

## 1. Host prerequisites

- Linux
- Docker Engine + Docker Compose v2
- NVIDIA driver + NVIDIA Container Toolkit
- Network route to HSR

```bash
docker --version
docker compose version
nvidia-smi
```

Verified environment (2026-02-20):

- OS: Ubuntu 24.04.3 LTS
- GPU: NVIDIA GeForce RTX 5070 Ti
- NVIDIA driver: 580.126.09
- Docker: 29.0.1
- Docker Compose: v2.40.3

## 2. Required environment variables

```bash
export TEST_MODE=false
export HSR_IP=100.119.167.94
export ROS_MASTER_URI=http://100.119.167.94:11311
export ROS_IP=<YOUR_HOST_IP>
export POLICY_CHECKPOINT_PATH=/abs/path/to/checkpoint_dir
export POLICY_SERVER_HOST=127.0.0.1
export POLICY_SERVER_PORT=8000
export POLICY_SERVER_API_KEY=
export POLICY_CACHE_DIR=$PWD/.docker_cache/policy_cache
export HF_CACHE_DIR=$PWD/.docker_cache/hf
export ROSBAG_DIR=$PWD/datasets/rosbags
```

## 3. Optional policy-specific variables

Set policy-specific variables only when your server implementation requires them.

OpenPI example:

```bash
export POLICY_CONFIG_NAME=<openpi_config_name>
export POLICY_DEFAULT_PROMPT=
export POLICY_RECORD_DIR=
export POLICY_PYTORCH_DEVICE=cuda
```

## 4. Start and verify containers

```bash
./RUN-DOCKER-CONTAINER.sh up
./RUN-DOCKER-CONTAINER.sh logs policy_server
./RUN-DOCKER-CONTAINER.sh logs hsr_client
```

## 5. Run deploy launch

1. Enter client shell:

```bash
./RUN-DOCKER-CONTAINER.sh shell
```

2. Launch inside client container:

```bash
roslaunch hsr_policy_client hsr_policy_client.launch test_mode:=false
```

Optional pre-check in the same shell:

```bash
roslaunch hsr_policy_client hsr_policy_client.launch
```

This pre-check runs in `test_mode:=true` by default and uses synthetic random observations.

## 6. Update language instruction at runtime

From a shell with ROS environment loaded:

```bash
rosservice call /hsr_policy_client/update_instruction "message: 'Pick up the coffee bottle on the right'"
```

## 7. Restart rules

Run `down` and `up` after changing:

- `POLICY_CHECKPOINT_PATH`
- policy-specific server environment values
- client-side source under `deploy/hsr_policy_client/*` (image rebuild required)

## 8. Troubleshooting

- `required variable ROS_MASTER_URI is missing`: set `ROS_MASTER_URI`
- `Container 'airoa_hsr_client' is not running`: run `./RUN-DOCKER-CONTAINER.sh up`
- `POLICY_CHECKPOINT_PATH is missing`: set `POLICY_CHECKPOINT_PATH`
- client waits for WebSocket server: check `./RUN-DOCKER-CONTAINER.sh logs policy_server`
- `unable to contact ROS master`: check network route, `ROS_MASTER_URI`, and `ROS_IP`
- `_tkinter.TclError: couldn't connect to display`: run from a graphical session (`DISPLAY` is valid), or reconnect with `ssh -Y`
- `Action not executed. reason=control_mode=...`: ensure `/control_mode` is `auto` (default is now `auto` when topic is absent)

## 9. Stop

```bash
./RUN-DOCKER-CONTAINER.sh down
```
