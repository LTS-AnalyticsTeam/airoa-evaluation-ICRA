#!/usr/bin/env bash
set -euo pipefail

# Some ROS/TMC setup scripts reference optional vars without guarding for nounset.
set +u
source /opt/ros/noetic/setup.bash
source /root/catkin_ws/devel/setup.bash
set -u

export PATH="/home/policy/.venv/bin:${PATH}"
export PYTHONPATH="/root/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:${PYTHONPATH:-}"

exec "$@"
