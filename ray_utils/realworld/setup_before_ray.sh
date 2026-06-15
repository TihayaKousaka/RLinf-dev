#!/bin/bash

export CURRENT_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$CURRENT_PATH"))
export PYTHONPATH=$REPO_PATH:$PYTHONPATH

# Modify these environment variables as needed
export RLINF_NODE_RANK=-1 # Change this to the appropriate node rank if using multiple nodes
# Optional: set this only when you need to pin communication to a specific NIC.
# Example NIC names: eth0, eno1, enp134s0f0.
export RLINF_COMM_NET_DEVICES="<nic_name>"

# If you are using the docker image, change this to source switch_env franka-<version>, e.g., switch_env franka-0.15.0
source <your_venv_path>/bin/activate # Source your virtual environment here

# Additionally source your own catkin workspace setup.bash if you are not installing franka_ros and serl_franka_controllers via the docker image or installation script
# source <your_catkin_ws>/devel/setup.bash
