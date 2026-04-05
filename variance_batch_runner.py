"""
Batch Experiment Runner for Bee Foraging Simulation

Author: NikTop
Date: 03/03/2026

Description:
This script runs repeated simulation experiments to collect quantitative data
on the evolution of social vs personal learning strategies.

It executes multiple independent runs (with different random seeds) and records:
- Population counts (social vs personal)
- Average rewards per strategy
- Evolution over generations

Results are saved as:
- CSV files for analysis
- Matplotlib plots for visualisation

Usage:
- Run directly to generate results:
    python batch_runner.py

- Output files:
    simulation_results.csv
    population_<mode>.png

Notes:
- Uses the Simulation class from bee_simulation.py
- Designed for controlled experiments with fixed parameters
- Does NOT render graphics (faster than visual simulation)
"""

import csv
import copy
import random
import statistics
import matplotlib.pyplot as plt

from bee_simulation import Simulation, CFG, generate_capacities


def get_post_evolution_counts(sim):
    social = sum(1 for b in sim.bees if b.strategy == "social")
    personal = len(sim.bees) - social
    return social, personal


def run_one_condition(cfg, mode, rich_count_high, repeats=10, generations=50):
    rows = []

    # make local config copy so we do not mutate the imported CFG
    local_cfg = copy.deepcopy(cfg)
    local_cfg["RICH_COUNT_HIGH"] = rich_count_high

    # compute actual reward variance for this condition
    # for stable mode, all flowers are rich, so variance should be 0
    if mode == "stable":
        caps = generate_capacities(local_cfg, "stable")
    else:
        caps = generate_capacities(local_cfg, "high")

    reward_variance = statistics.pvariance(caps)

    for repeat in range(repeats):
        # reproducible but different repeat/mode/condition
        if local_cfg.get("SEED") is not None:
            seed = (
                local_cfg["SEED"]
                + 1000 * repeat
                + 10000 * rich_count_high
                + (0 if mode == "stable" else 500000)
            )
            random.seed(seed)
        else:
            seed = None

        sim = Simulation(local_cfg, mode=mode)
        sim.reset_world()

        target_generation = generations

        while sim.generation < target_generation:
            prev_generation = sim.generation
            sim.update()

            if sim.generation > prev_generation:
                social, personal = get_post_evolution_counts(sim)

                rows.append({
                    "mode": mode,
                    "repeat": repeat,
                    "generation": sim.generation,
                    "rich_count_high": rich_count_high,
                    "reward_variance": reward_variance,
                    "social": social,
                    "personal": personal,
                    "social_ratio": social / len(sim.bees),
                    "personal_ratio": personal / len(sim.bees),
                    "social_reward": sim.last_generation_social_mean_reward,
                    "personal_reward": sim.last_generation_personal_mean_reward,
                    "seed": seed,
                })

        print(
            f"done mode={mode}, rich_count_high={rich_count_high}, "
            f"variance={reward_variance:.3f}, repeat={repeat+1}/{repeats}"
        )

    return rows


def save_csv(rows, filename):
    if not rows:
        print("No rows to save.")
        return

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_final_rows(rows):
    grouped = {}

    for row in rows:
        key = (row["mode"], row["rich_count_high"], row["reward_variance"], row["repeat"])
        if key not in grouped:
            grouped[key] = row
        else:
            if row["generation"] > grouped[key]["generation"]:
                grouped[key] = row

    return list(grouped.values())


def plot_final_social_vs_variance(final_rows, out_path="final_social_vs_variance.png"):
    # aggregate across repeats
    grouped = {}

    for row in final_rows:
        key = (row["mode"], row["rich_count_high"], row["reward_variance"])
        grouped.setdefault(key, []).append(row["social_ratio"])

    stable_points = []
    high_points = []

    for (mode, rich_count_high, reward_variance), vals in grouped.items():
        mean_social = sum(vals) / len(vals)
        point = {
            "variance": reward_variance,
            "mean_social_ratio": mean_social,
            "rich_count_high": rich_count_high
        }

        if mode == "stable":
            stable_points.append(point)
        else:
            high_points.append(point)

    stable_points.sort(key=lambda x: x["variance"])
    high_points.sort(key=lambda x: x["variance"])

    plt.figure(figsize=(9, 6))

    if stable_points:
        plt.plot(
            [p["variance"] for p in stable_points],
            [p["mean_social_ratio"] for p in stable_points],
            marker="o",
            label="Stable mode"
        )

    if high_points:
        plt.plot(
            [p["variance"] for p in high_points],
            [p["mean_social_ratio"] for p in high_points],
            marker="o",
            label="High mode"
        )

    plt.xlabel("Reward variance across flowers")
    plt.ylabel("Final social proportion")
    plt.title("Final social proportion vs reward variance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_final_population_ratio_vs_rich_count(final_rows, out_path="final_social_vs_richcount.png"):
    grouped = {}

    for row in final_rows:
        key = (row["mode"], row["rich_count_high"])
        grouped.setdefault(key, []).append(row["social_ratio"])

    stable_points = []
    high_points = []

    for (mode, rich_count_high), vals in grouped.items():
        mean_social = sum(vals) / len(vals)
        point = {
            "rich_count_high": rich_count_high,
            "mean_social_ratio": mean_social,
        }

        if mode == "stable":
            stable_points.append(point)
        else:
            high_points.append(point)

    stable_points.sort(key=lambda x: x["rich_count_high"])
    high_points.sort(key=lambda x: x["rich_count_high"])

    plt.figure(figsize=(9, 6))

    if stable_points:
        plt.plot(
            [p["rich_count_high"] for p in stable_points],
            [p["mean_social_ratio"] for p in stable_points],
            marker="o",
            label="Stable mode"
        )

    if high_points:
        plt.plot(
            [p["rich_count_high"] for p in high_points],
            [p["mean_social_ratio"] for p in high_points],
            marker="o",
            label="High mode"
        )

    plt.xlabel("Number of rich flowers")
    plt.ylabel("Final social proportion")
    plt.title("Final social proportion vs number of rich flowers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    local_cfg = copy.deepcopy(CFG)

    # make sure these match your visual experiment settings
    local_cfg["GENERATION_LENGTH"] = 5000

    # for controlled experiment, use constant distance
    # local_cfg["FLOWER_DISTANCE_MODE"] = "constant"
    # local_cfg["FLOWER_DISTANCE_CONSTANT"] = 250

    repeats = 1
    generations = 50

    # gradient from no variance to very high variance
    rich_counts_to_test = [24, 16, 12, 8, 4, 2]

    all_rows = []

    for rich_count_high in rich_counts_to_test:
        # stable mode is included for completeness,
        # but its variance is always basically 0
        all_rows.extend(
            run_one_condition(
                cfg=local_cfg,
                mode="stable",
                rich_count_high=rich_count_high,
                repeats=repeats,
                generations=generations
            )
        )

        all_rows.extend(
            run_one_condition(
                cfg=local_cfg,
                mode="high",
                rich_count_high=rich_count_high,
                repeats=repeats,
                generations=generations
            )
        )

    save_csv(all_rows, "variance_sweep_all_generations.csv")

    final_rows = aggregate_final_rows(all_rows)
    save_csv(final_rows, "variance_sweep_final_only.csv")

    plot_final_social_vs_variance(final_rows, "final_social_vs_variance.png")
    plot_final_population_ratio_vs_rich_count(final_rows, "final_social_vs_richcount.png")

    print("Done. Saved:")
    print("- variance_sweep_all_generations.csv")
    print("- variance_sweep_final_only.csv")
    print("- final_social_vs_variance.png")
    print("- final_social_vs_richcount.png")


if __name__ == "__main__":
    main()