#!/bin/bash

# Simple script to run training on DMC tasks
# Usage: ./run_DMC.sh
# Or with environment variables: TASK=dmc_walker_walk METHOD=ne_dreamer SEED=42 ./run_DMC.sh

# ==== METHOD ====
# Available options: r2dreamer, ne_dreamer, dreamerpro, dreamer
METHOD=${METHOD:-r2dreamer}

# ==== Seed ====
SEED=${SEED:-0}

# ==== Task ====
# DMC tasks available:
# dmc_acrobot_swingup, dmc_ball_in_cup_catch, dmc_cartpole_balance,
# dmc_cartpole_balance_sparse, dmc_cartpole_swingup, dmc_cartpole_swingup_sparse,
# dmc_cheetah_run, dmc_finger_spin, dmc_finger_turn_easy, dmc_finger_turn_hard,
# dmc_hopper_hop, dmc_hopper_stand, dmc_pendulum_swingup, dmc_quadruped_run,
# dmc_quadruped_walk, dmc_reacher_easy, dmc_reacher_hard, dmc_walker_run,
# dmc_walker_stand, dmc_walker_walk
task=${TASK:-dmc_walker_walk}

# ==== Model hyperparameters ====
imag_horizon=${IMAG_HORIZON:-15}
horizon_discount=${HORIZON_DISCOUNT:-0.85}

# Validate imag_horizon is an integer
if ! [[ "$imag_horizon" =~ ^[0-9]+$ ]]; then
    echo "Error: imag_horizon ($imag_horizon) is not an integer."
    exit 1
fi

# ==== Run training ====
echo "Running ${METHOD} on ${task} with seed=${SEED}"

python3 train.py \
    env.task=$task \
    model.rep_loss=${METHOD} \
    model.imag_horizon=${imag_horizon} \
    model.horizon_discount=${horizon_discount} \
    seed=$SEED \
    logdir=logdir/${METHOD}_${task#dmc_}_s${SEED}
