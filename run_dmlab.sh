#!/bin/bash

# Simple script to run training on DMLab30 tasks
# Usage: ./run_dmlab.sh
# Or with environment variables: TASK=dmlab_rooms_collect_good_objects_train METHOD=ne_dreamer SEED=42 ./run_dmlab.sh
#
# Note: DMLab requires additional setup. See requirements_dmlab.txt for dependencies.

# ==== METHOD ====
# Available options: r2dreamer, ne_dreamer, dreamerpro, dreamer
METHOD=${METHOD:-r2dreamer}

# ==== Seed ====
SEED=${SEED:-0}

# ==== Task ====
# DMLab30 tasks available:
# dmlab_rooms_collect_good_objects_train, dmlab_rooms_exploit_deferred_effects_train,
# dmlab_rooms_select_nonmatching_object, dmlab_rooms_watermaze, dmlab_rooms_keys_doors_puzzle,
# dmlab_natlab_fixed_large_map, dmlab_natlab_varying_map_regrowth, dmlab_natlab_varying_map_randomized,
# dmlab_skymaze_irreversible_path_hard, dmlab_skymaze_irreversible_path_varied,
# dmlab_psychlab_arbitrary_visuomotor_mapping, dmlab_psychlab_continuous_recognition,
# dmlab_psychlab_sequential_comparison, dmlab_psychlab_visual_search,
# dmlab_explore_object_locations_small, dmlab_explore_object_locations_large,
# dmlab_explore_obstructed_goals_small, dmlab_explore_obstructed_goals_large,
# dmlab_explore_goal_locations_small, dmlab_explore_goal_locations_large,
# dmlab_explore_object_rewards_few, dmlab_explore_object_rewards_many,
# dmlab_lasertag_one_opponent_small, dmlab_lasertag_three_opponents_small,
# dmlab_lasertag_one_opponent_large, dmlab_lasertag_three_opponents_large,
# dmlab_language_select_described_object, dmlab_language_select_located_object,
# dmlab_language_execute_random_task, dmlab_language_answer_quantitative_question
task=${TASK:-dmlab_rooms_collect_good_objects_train}

# ==== Model hyperparameters ====
imag_horizon=${IMAG_HORIZON:-15}
horizon_discount=${HORIZON_DISCOUNT:-0.85}
action_set=${ACTION_SET:-default}

# Validate imag_horizon is an integer
if ! [[ "$imag_horizon" =~ ^[0-9]+$ ]]; then
    echo "Error: imag_horizon ($imag_horizon) is not an integer."
    exit 1
fi

# ==== Run training ====
echo "Running ${METHOD} on ${task} with seed=${SEED}"

python3 train.py \
    env=dmlab_vision \
    env.task=$task \
    env.action_set=$action_set \
    model.rep_loss=${METHOD} \
    model.imag_horizon=${imag_horizon} \
    model.horizon_discount=${horizon_discount} \
    seed=$SEED \
    logdir=logdir/${METHOD}_${task#dmlab_}_s${SEED}
