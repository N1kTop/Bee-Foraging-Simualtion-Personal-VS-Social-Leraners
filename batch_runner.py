import random
import csv
import math
import os
import matplotlib.pyplot as plt

# import your simulation directly
from bee_simulation import Simulation, CFG


def run_batch(cfg, modes=("stable", "high"), repeats=10, generations=60):
    results = []

    for mode in modes:
        for repeat in range(repeats):
            sim = Simulation(cfg, mode = mode)
            sim.reset_world()

            target_generation = generations

            while sim.generation < target_generation:
                previous_generation = sim.generation
                sim.update()

                # record once when a new generation is completed
                if sim.generation > previous_generation:
                    social = sim.last_generation_social_count
                    personal = sim.last_generation_personal_count

                    results.append({
                        "mode": mode,
                        "repeat": repeat,
                        "generation": sim.generation,
                        "social": social,
                        "personal": personal,
                        "social_reward": sim.last_generation_social_mean_reward,
                        "personal_reward": sim.last_generation_personal_mean_reward
                    })

                    print(f"mode={mode}, repeat={repeat+1}/{repeats}, generation={sim.generation}/{target_generation}")

    return results

def save_csv(results, filename="results.csv"):
    keys = results[0].keys()

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def plot_results(results):
    import collections

    grouped = collections.defaultdict(lambda: {"social": [], "personal": []})

    for r in results:
        key = (r["mode"], r["generation"])
        grouped[key]["social"].append(r["social"])
        grouped[key]["personal"].append(r["personal"])

    modes = set(r["mode"] for r in results)

    for mode in modes:
        gens = sorted(set(r["generation"] for r in results if r["mode"] == mode))

        social_avg = []
        personal_avg = []

        for g in gens:
            vals = grouped[(mode, g)]
            social_avg.append(sum(vals["social"]) / len(vals["social"]))
            personal_avg.append(sum(vals["personal"]) / len(vals["personal"]))

        plt.figure()
        plt.plot(gens, social_avg, label="Social")
        plt.plot(gens, personal_avg, label="Personal")
        plt.title(f"Population over time ({mode})")
        plt.xlabel("Generation")
        plt.ylabel("Population")
        plt.legend()
        plt.savefig(f"population_{mode}.png")


if __name__ == "__main__":
    CFG["GENERATION_LENGTH"] = 5000
    results = run_batch(CFG, repeats=10, generations=50)

    save_csv(results, "simulation_results.csv")
    plot_results(results)

    print("Done. CSV + plots saved.")