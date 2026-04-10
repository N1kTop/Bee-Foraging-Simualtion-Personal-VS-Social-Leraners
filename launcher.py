"""
Preset Launcher for Bee Foraging Simulation

Author: NikTop
Date: 03/03/2026

Description:
This script provides a convenient interface for running the simulation
with predefined configuration presets.

Each preset represents a different experimental condition, such as:
- Stable environment (equal rewards)
- High-variance environment
- Dynamic environments with depletion and noise
- Randomised spatial layouts

Usage:
    python launcher.py

The user will be prompted to select a preset.

Features:
- Automatically builds configuration from base CFG
- Supports multiple predefined experimental scenarios
- Integrates with interactive simulation (pygame)

Notes:
- Presets override selected parameters from bee_simulation.py
- START_MODE controls whether the simulation begins in "stable" or "high" mode
- Designed for demonstrations and presentations
"""

import copy
from bee_simulation import CFG, main as run_simulation


PRESETS = {
    "NORMAL_HIGH": {
        "WIDTH": 1600,
        "HEIGHT": 900,
        "WORLD_WIDTH": 1100,
        "PANEL_WIDTH": 500,
        "FPS": 60,
        "BACKGROUND": (245, 245, 245),
        "WORLD_MARGIN": 80,
        "NUMBER_OF_FLOWERS": 24,
        "RICH_COUNT_HIGH": 4,
        "MEAN_REWARD_PER_FLOWER": 10.0,
        "RICH_VS_POOR_REWARD_RATIO": 5.0,
        "TOTAL_BEES": 80,
        "BEES_REPLACED_PER_GENERATION": 20,
        "STARTING_BEE_RATIO": 0.5,
        "BEE_SPEED": 1.8,
        "BEE_RADIUS": 4,
        "PERSONAL_EPSILON": 0.15,
        "PERSONAL_ALPHA": 0.10,
        "SOCIAL_OBSERVATION_NOISE": 0.0,
        "SOCIAL_MEMORY_LENGTH": 200,
        "ENABLE_DEPLETION": False,
        "HARVEST_STRENGTH": 0.25,
        "REGEN_RATE": 0.00,
        "ENABLE_NOISE": False,
        "NOISE_STD": 0.02,
        "CROWDING_PARAMETER": 0.5,
        "INHERIT_PARENT_MEMORY": False,
        "FLOWER_MIN_RADIUS": 10,
        "FLOWER_MAX_RADIUS": 22,
        "SHOW_TARGET_LINES": False,
        "SHOW_FLOWER_LABELS": True,
        "SHOW_DEBUG_RICH": False,
        "SHOW_FLOWER_ID": True,
        "FLOWER_DISTANCE_MODE": "constant",
        "FLOWER_DISTANCE_CONSTANT": 250,
        "FLOWER_DISTANCE_MIN": 180,
        "FLOWER_DISTANCE_MAX": 320,
        "SIMULATION_SPEED": 1,
        "GENERATION_LENGTH": 5000,
        "MUTATION_RATE": 0.00,
        "RESET_FLOWERS_EACH_GENERATION": True,
        "GRAPH_MAX_POINTS": 200,
        "START_MODE": "high"
    },

    "NORMAL_STABLE": {
        "WIDTH": 1600,
        "HEIGHT": 900,
        "WORLD_WIDTH": 1100,
        "PANEL_WIDTH": 500,
        "FPS": 60,
        "BACKGROUND": (245, 245, 245),
        "WORLD_MARGIN": 80,
        "NUMBER_OF_FLOWERS": 24,
        "RICH_COUNT_HIGH": 0,
        "MEAN_REWARD_PER_FLOWER": 10.0,
        "RICH_VS_POOR_REWARD_RATIO": 5.0,
        "TOTAL_BEES": 80,
        "BEES_REPLACED_PER_GENERATION": 20,
        "STARTING_BEE_RATIO": 0.5,
        "BEE_SPEED": 1.8,
        "BEE_RADIUS": 4,
        "PERSONAL_EPSILON": 0.15,
        "PERSONAL_ALPHA": 0.10,
        "SOCIAL_OBSERVATION_NOISE": 0.0,
        "SOCIAL_MEMORY_LENGTH": 200,
        "ENABLE_DEPLETION": False,
        "HARVEST_STRENGTH": 0.25,
        "REGEN_RATE": 0.00,
        "ENABLE_NOISE": True,
        "NOISE_STD": 0.02,
        "CROWDING_PARAMETER": 0.5,
        "INHERIT_PARENT_MEMORY": False,
        "FLOWER_MIN_RADIUS": 10,
        "FLOWER_MAX_RADIUS": 22,
        "SHOW_TARGET_LINES": False,
        "SHOW_FLOWER_LABELS": True,
        "SHOW_DEBUG_RICH": False,
        "SHOW_FLOWER_ID": True,
        "FLOWER_DISTANCE_MODE": "constant",
        "FLOWER_DISTANCE_CONSTANT": 250,
        "FLOWER_DISTANCE_MIN": 180,
        "FLOWER_DISTANCE_MAX": 320,
        "SIMULATION_SPEED": 1,
        "GENERATION_LENGTH": 5000,
        "MUTATION_RATE": 0.00,
        "RESET_FLOWERS_EACH_GENERATION": True,
        "GRAPH_MAX_POINTS": 200,
        "START_MODE": "stable"
    },

    "DYNAMIC_HIGH": {
        "WIDTH": 1600,
        "HEIGHT": 900,
        "WORLD_WIDTH": 1100,
        "PANEL_WIDTH": 500,
        "FPS": 60,
        "BACKGROUND": (245, 245, 245),
        "WORLD_MARGIN": 80,
        "NUMBER_OF_FLOWERS": 24,
        "RICH_COUNT_HIGH": 4,
        "MEAN_REWARD_PER_FLOWER": 10.0,
        "RICH_VS_POOR_REWARD_RATIO": 5.0,
        "TOTAL_BEES": 80,
        "BEES_REPLACED_PER_GENERATION": 20,
        "STARTING_BEE_RATIO": 0.5,
        "BEE_SPEED": 1.8,
        "BEE_RADIUS": 4,
        "PERSONAL_EPSILON": 0.15,
        "PERSONAL_ALPHA": 0.10,
        "SOCIAL_OBSERVATION_NOISE": 0.0,
        "SOCIAL_MEMORY_LENGTH": 200,
        "ENABLE_DEPLETION": True,
        "HARVEST_STRENGTH": 0.25,
        "REGEN_RATE": 0.03,
        "ENABLE_NOISE": False,
        "NOISE_STD": 0.02,
        "CROWDING_PARAMETER": 0.25,
        "TRAVEL_COST_PER_PIXEL": 0.01,
        "INHERIT_PARENT_MEMORY": False,
        "FLOWER_MIN_RADIUS": 10,
        "FLOWER_MAX_RADIUS": 22,
        "SHOW_TARGET_LINES": False,
        "SHOW_FLOWER_LABELS": True,
        "SHOW_DEBUG_RICH": False,
        "SHOW_FLOWER_ID": True,
        "FLOWER_DISTANCE_MODE": "range",
        "FLOWER_DISTANCE_CONSTANT": 250,
        "FLOWER_DISTANCE_MIN": 180,
        "FLOWER_DISTANCE_MAX": 320,
        "SIMULATION_SPEED": 1,
        "GENERATION_LENGTH": 5000,
        "MUTATION_RATE": 0.02,
        "RESET_FLOWERS_EACH_GENERATION": True,
        "GRAPH_MAX_POINTS": 200,
        "START_MODE": "high"
    },
    
    "dynamic_test": {
        "WIDTH": 1600,
        "HEIGHT": 900,
        "WORLD_WIDTH": 1100,
        "PANEL_WIDTH": 500,
        "FPS": 60,
        "BACKGROUND": (245, 245, 245),
        "WORLD_MARGIN": 80,
        "NUMBER_OF_FLOWERS": 24,
        "RICH_COUNT_HIGH": 4,
        "MEAN_REWARD_PER_FLOWER": 10.0,
        "RICH_VS_POOR_REWARD_RATIO": 5.0,
        "TOTAL_BEES": 80,
        "BEES_REPLACED_PER_GENERATION": 20,
        "STARTING_BEE_RATIO": 0.5,
        "BEE_SPEED": 1.8,
        "BEE_RADIUS": 4,
        "PERSONAL_EPSILON": 0.15,
        "PERSONAL_ALPHA": 0.10,
        "SOFTMAX_TAU": 0.02,
        "SOCIAL_OBSERVATION_NOISE": 0.0,
        "SOCIAL_MEMORY_LENGTH": 200,
        "ENABLE_DEPLETION": False,
        "HARVEST_STRENGTH": 0.0,
        "REGEN_RATE": 0.0,
        "ENABLE_NOISE": False,
        "NOISE_STD": 0.0,
        "CROWDING_PARAMETER": 0.4,
        "EFFICIENCY_SCALE": 50.0,
        "INHERIT_PARENT_MEMORY": False,
        "FLOWER_MIN_RADIUS": 10,
        "FLOWER_MAX_RADIUS": 22,
        "SHOW_TARGET_LINES": False,
        "SHOW_FLOWER_LABELS": True,
        "SHOW_DEBUG_RICH": False,
        "SHOW_FLOWER_ID": True,
        "FLOWER_DISTANCE_MODE": "range",
        "FLOWER_DISTANCE_CONSTANT": 250,
        "FLOWER_DISTANCE_MIN": 180,
        "FLOWER_DISTANCE_MAX": 320,
        "SIMULATION_SPEED": 1,
        "GENERATION_LENGTH": 5000,
        "MUTATION_RATE": 0.01,
        "RESET_FLOWERS_EACH_GENERATION": True,
        "GRAPH_MAX_POINTS": 200,
        "START_MODE": "high"
    },

    "paper_stable": {
        "RICH_COUNT_HIGH": 4,
        "FLOWER_DISTANCE_MODE": "constant",
        "FLOWER_DISTANCE_CONSTANT": 250,
        "ENABLE_DEPLETION": False,
        "HARVEST_STRENGTH": 0.0,
        "REGEN_RATE": 0.0,
        "ENABLE_NOISE": False,
        "NOISE_STD": 0.0,
        "CROWDING_PARAMETER": 0.4,
        "INHERIT_PARENT_MEMORY": False,
        "MUTATION_RATE": 0.01,
        "SEED": 42,
        "START_MODE": "stable",
    },

    "paper_high_variance": {
        "RICH_COUNT_HIGH": 4,
        "FLOWER_DISTANCE_MODE": "constant",
        "FLOWER_DISTANCE_CONSTANT": 250,
        "ENABLE_DEPLETION": False,
        "HARVEST_STRENGTH": 0.0,
        "REGEN_RATE": 0.0,
        "ENABLE_NOISE": False,
        "NOISE_STD": 0.0,
        "CROWDING_PARAMETER": 0.4,
        "INHERIT_PARENT_MEMORY": False,
        "MUTATION_RATE": 0.01,
        "SEED": 42,
        "START_MODE": "high",
    },

    "dynamic_medium": {
        "RICH_COUNT_HIGH": 4,
        "FLOWER_DISTANCE_MODE": "constant",
        "FLOWER_DISTANCE_CONSTANT": 250,
        "ENABLE_DEPLETION": True,
        "HARVEST_STRENGTH": 0.15,
        "REGEN_RATE": 0.02,
        "ENABLE_NOISE": True,
        "NOISE_STD": 0.015,
        "CROWDING_PARAMETER": 0.4,
        "INHERIT_PARENT_MEMORY": False,
        "MUTATION_RATE": 0.01,
        "SEED": 42,
        "START_MODE": "high",
    },

    "dynamic_high": {
        "RICH_COUNT_HIGH": 4,
        "FLOWER_DISTANCE_MODE": "constant",
        "FLOWER_DISTANCE_CONSTANT": 250,
        "ENABLE_DEPLETION": True,
        "HARVEST_STRENGTH": 0.30,
        "REGEN_RATE": 0.03,
        "ENABLE_NOISE": True,
        "NOISE_STD": 0.03,
        "CROWDING_PARAMETER": 0.4,
        "INHERIT_PARENT_MEMORY": False,
        "MUTATION_RATE": 0.01,
        "SEED": 42,
        "START_MODE": "high",
    },

    "random_distance_demo": {
        "RICH_COUNT_HIGH": 4,
        "FLOWER_DISTANCE_MODE": "range",
        "FLOWER_DISTANCE_MIN": 180,
        "FLOWER_DISTANCE_MAX": 320,
        "ENABLE_DEPLETION": True,
        "HARVEST_STRENGTH": 0.15,
        "REGEN_RATE": 0.02,
        "ENABLE_NOISE": True,
        "NOISE_STD": 0.015,
        "CROWDING_PARAMETER": 0.4,
        "INHERIT_PARENT_MEMORY": False,
        "MUTATION_RATE": 0.01,
        "SEED": 42,
        "START_MODE": "high",
    },
}


def build_cfg(preset_name):
    cfg = copy.deepcopy(CFG)
    preset = PRESETS[preset_name]
    cfg.update({k: v for k, v in preset.items() if k != "START_MODE"})
    return cfg, preset["START_MODE"]


if __name__ == "__main__":
    print("Available presets:")
    for name in PRESETS:
        print("-", name)

    choice = input("Enter preset name: ").strip()

    if choice not in PRESETS:
        print(f"Unknown preset: {choice}")
    else:
        cfg, start_mode = build_cfg(choice)
        run_simulation(cfg=cfg, start_mode=start_mode)