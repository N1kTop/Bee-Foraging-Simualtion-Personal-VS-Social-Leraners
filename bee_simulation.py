"""
Bee Foraging Simulation with Social vs Personal Learning

Author: NikTop
Date: 03/03/2026

Description:
This project implements an agent-based simulation of bee foraging behaviour,
comparing social learning (copying others) and personal learning (reinforcement-based).

Based on:
    Smolla et al. (2016)
    "Copy-when-uncertain: bumblebees rely on social information when rewards are highly variable"
    Biology Letters, 12: 20160188
    DOI: https://doi.org/10.1098/rsbl.2016.0188

This simulation implements the agent-based model described in the paper.
But it differes from the original version in the paper and extends it by:
- Modelling resource depletion and regeneration
- Adding stochastic environmental noise
- Supporting dynamic environments with configurable variability
- Providing both interactive visualisation (Pygame) and batch experimentation tools

Key Features:
- Two learning strategies: social vs personal
- Evolutionary selection based on foraging efficiency
- Configurable environmental dynamics (depletion, noise)
- Data collection via batch runners (CSV + plots)
- Real-time visualisation with adjustable simulation speed

Usage:
- Run this file directly for interactive simulation
- Use batch_runner.py or variance_batch_runner.py for experiments
- Use launcher.py for better config preset manager

Notes:
- All parameters are controlled via the CFG dictionary
- Results may vary depending on random seed

"""

import copy
import math
import random
import pygame
from dataclasses import dataclass, field
from typing import Optional


# Central configuration for simulation behaviour, rendering, and tuning parameters
CFG = {
    # --- Window ---
    "WIDTH": 1600,
    "HEIGHT": 900,
    "WORLD_WIDTH": 1100,                # simulation area
    "PANEL_WIDTH": 500,                 # analytics panel
    "FPS": 60,
    "BACKGROUND": (245, 245, 245),

    # --- Environment ---
    "WORLD_MARGIN": 80,                 # spacing from edges when placing flowers
    "NUMBER_OF_FLOWERS": 24,
    "RICH_COUNT_HIGH": 4,               # number of high-reward flowers in "high" mode
    "MEAN_REWARD_PER_FLOWER": 10.0,
    "RICH_VS_POOR_REWARD_RATIO": 5.0,   # how much better rich flowers are

    # --- Bees ---
    "TOTAL_BEES": 80,
    "BEES_REPLACED_PER_GENERATION": 20, # evolutionary turnover
    "STARTING_BEE_RATIO": 0.5,          # proportion of personal learners
    "BEE_SPEED": 1.8,
    "BEE_RADIUS": 4,

    # Personal Learners ---
    "PERSONAL_EPSILON": 0.15,           # exploration probability
    "PERSONAL_ALPHA": 0.10,             # learning rate
    "SOFTMAX_TAU": 0.02,                # softmax temperature (lower = more greedy)

    # --- Social Learners ---
    "SOCIAL_OBSERVATION_NOISE": 0.0,    # noise in observing other bees
    "SOCIAL_MEMORY_LENGTH": 200,        # how long visit history is remembered

    # --- Resource dynamics ---
    "ENABLE_DEPLETION": False,
    "HARVEST_STRENGTH": 0.0,            # how much nectar is removed per visit
    "REGEN_RATE": 0.0,                  # how fast flowers recover
    "ENABLE_NOISE": False,
    "NOISE_STD": 0.0,                   # stochastic noise in resource levels

    # --- Reward shaping ---
    "CROWDING_PARAMETER": 0.4,          # penalty for many bees on same flower
    "EFFICIENCY_SCALE": 50.0,           # scales reward per distance (learning signal)

    # --- Extra ---
    "INHERIT_PARENT_MEMORY": False,     # whether offspring inherit learned Q-values

    # --- Visual settings ---
    "FLOWER_MIN_RADIUS": 10,
    "FLOWER_MAX_RADIUS": 22,
    "SHOW_TARGET_LINES": False,
    "SHOW_FLOWER_LABELS": True,
    "SHOW_DEBUG_RICH": False,
    "SHOW_FLOWER_ID": True,

    # --- Flower placement ---
    "FLOWER_DISTANCE_MODE": "range",    # "random", "constant", or "range"
    "FLOWER_DISTANCE_CONSTANT": 250,
    "FLOWER_DISTANCE_MIN": 180,
    "FLOWER_DISTANCE_MAX": 320,

    # --- Simulation control ---
    "SIMULATION_SPEED": 1,
    "GENERATION_LENGTH": 5000,
    "MUTATION_RATE": 0.01,
    "RESET_FLOWERS_EACH_GENERATION": True,

    # --- Graphing ---
    "GRAPH_MAX_POINTS": 200,

    # --- Random Seed ---
    "SEED": 42,
}


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def weighted_choice(indices, weights):
    indices = list(indices)
    total = sum(weights)

    if total <= 0:
        return random.choice(indices)

    r = random.uniform(0, total)
    acc = 0.0
    for idx, w in zip(indices, weights):
        acc += w
        if r <= acc:
            return idx

    return indices[-1]


def colour_lerp(a, b, t):
    t = clamp(t, 0.0, 1.0)
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def trim_history(lst, max_points):
    if len(lst) > max_points:
        del lst[:-max_points]


@dataclass
class Flower:
    idx: int
    x: float
    y: float
    capacity: float
    stock: float
    is_rich: bool

    def radius(self, cfg):
        capacities = cfg["_all_capacities"]
        cmin = min(capacities)
        cmax = max(capacities)

        if abs(cmax - cmin) < 1e-9:
            return (cfg["FLOWER_MIN_RADIUS"] + cfg["FLOWER_MAX_RADIUS"]) // 2

        t = (self.capacity - cmin) / (cmax - cmin)
        return int(cfg["FLOWER_MIN_RADIUS"] + t * (cfg["FLOWER_MAX_RADIUS"] - cfg["FLOWER_MIN_RADIUS"]))

    def stock_fraction(self):
        if self.capacity <= 0:
            return 0.0
        return self.stock / self.capacity


@dataclass
class Bee:
    strategy: str
    x: float
    y: float
    speed: float
    reward_total: float = 0.0
    target_flower_idx: Optional[int] = None
    q_values: list[float] = field(default_factory=list)
    epsilon: float = 0.1
    alpha: float = 0.2

    state: str = "at_base"
    trip_distance: float = 0.0
    trip_nectar_reward: float = 0.0

    def choose_target(self, flowers, social_signal, cfg):
        # Personal bees learn from their own past reward estimates
        if self.strategy == "personal":
            if random.random() < self.epsilon:
                self.target_flower_idx = random.randrange(len(flowers))
                return

            tau = max(cfg["SOFTMAX_TAU"], 1e-6)
            max_q = max(self.q_values)
            shifted_q = [q - max_q for q in self.q_values]
            weights = [math.exp(q / tau) for q in shifted_q]
            self.target_flower_idx = weighted_choice(range(len(flowers)), weights)
            return

        # Social bees choose using the recent visit history of the colony
        indices = list(range(len(flowers)))
        noisy_counts = []

        for c in social_signal:
            scaled = c ** 1.5
            noisy = max(0.0, scaled + random.gauss(0, cfg["SOCIAL_OBSERVATION_NOISE"]))
            noisy_counts.append(noisy)

        if sum(noisy_counts) <= 0:
            self.target_flower_idx = random.randrange(len(flowers))
        else:
            self.target_flower_idx = weighted_choice(indices, noisy_counts)

    def move_towards_point(self, tx, ty):
        dx = tx - self.x
        dy = ty - self.y
        d = math.hypot(dx, dy)

        if d <= self.speed or d == 0:
            moved = d
            self.x = tx
            self.y = ty
            self.trip_distance += moved
            return True

        self.x += self.speed * dx / d
        self.y += self.speed * dy / d
        self.trip_distance += self.speed
        return False

    def update_personal_memory(self, flower_idx, efficiency):
        old = self.q_values[flower_idx]
        self.q_values[flower_idx] = old + self.alpha * (efficiency - old)

    def reset_for_new_generation(self, base_x, base_y, flowers, social_signal, cfg):
        self.x = base_x + random.randint(-10, 10)
        self.y = base_y + random.randint(-10, 10)
        self.reward_total = 0.0
        self.trip_distance = 0.0
        self.trip_nectar_reward = 0.0
        self.target_flower_idx = None
        self.state = "at_base"
        self.choose_target(flowers, social_signal, cfg)
        self.state = "to_flower"


def generate_capacities(cfg, mode):
    # In "stable" mode, all flowers are rich
    # In "high" mode, only a subset are rich and the rest are poor
    n = cfg["NUMBER_OF_FLOWERS"]
    rich_count = n if mode == "stable" else cfg["RICH_COUNT_HIGH"]

    total_reward = n * cfg["MEAN_REWARD_PER_FLOWER"]
    ratio = cfg["RICH_VS_POOR_REWARD_RATIO"]

    denominator = rich_count + (n - rich_count) / ratio
    rich_reward = total_reward / denominator
    poor_reward = rich_reward / ratio

    caps = [rich_reward] * rich_count + [poor_reward] * (n - rich_count)
    random.shuffle(caps)
    cfg["_all_capacities"] = caps[:]
    return caps


def create_flowers(cfg, mode):
    caps = generate_capacities(cfg, mode)
    flowers = []

    w = cfg["WORLD_WIDTH"]
    h = cfg["HEIGHT"]
    m = cfg["WORLD_MARGIN"]

    base_x = cfg["WORLD_WIDTH"] // 2
    base_y = cfg["HEIGHT"] // 2

    max_cap = max(caps)
    distance_mode = cfg["FLOWER_DISTANCE_MODE"]

    if distance_mode == "constant":
        radius = max(40, cfg["FLOWER_DISTANCE_CONSTANT"])
        angle_offset = random.uniform(0, 2 * math.pi)

        for i, cap in enumerate(caps):
            angle = angle_offset + (2 * math.pi * i / len(caps))
            x = base_x + radius * math.cos(angle)
            y = base_y + radius * math.sin(angle)

            x = clamp(x, m, w - m)
            y = clamp(y, m, h - m)

            flowers.append(
                Flower(
                    idx=i,
                    x=x,
                    y=y,
                    capacity=cap,
                    stock=cap,
                    is_rich=(cap == max_cap),
                )
            )

    elif distance_mode == "range":
        rmin = min(cfg["FLOWER_DISTANCE_MIN"], cfg["FLOWER_DISTANCE_MAX"])
        rmax = max(cfg["FLOWER_DISTANCE_MIN"], cfg["FLOWER_DISTANCE_MAX"])
        rmin = max(40, rmin)
        rmax = max(rmin, rmax)

        angle_offset = random.uniform(0, 2 * math.pi)

        for i, cap in enumerate(caps):
            angle = angle_offset + (2 * math.pi * i / len(caps))
            radius = random.uniform(rmin, rmax)

            x = base_x + radius * math.cos(angle)
            y = base_y + radius * math.sin(angle)

            x = clamp(x, m, w - m)
            y = clamp(y, m, h - m)

            flowers.append(
                Flower(
                    idx=i,
                    x=x,
                    y=y,
                    capacity=cap,
                    stock=cap,
                    is_rich=(cap == max_cap),
                )
            )

    else:
        for i, cap in enumerate(caps):
            while True:
                x = random.randint(m, w - m)
                y = random.randint(m, h - m)
                ok = True

                for f in flowers:
                    if distance(x, y, f.x, f.y) < 55:
                        ok = False
                        break

                if ok:
                    break

            flowers.append(
                Flower(
                    idx=i,
                    x=x,
                    y=y,
                    capacity=cap,
                    stock=cap,
                    is_rich=(cap == max_cap),
                )
            )

    return flowers


def create_bees(cfg, flowers, base_x, base_y):
    bees = []
    total = cfg["TOTAL_BEES"]
    personal_count = int(total * cfg["STARTING_BEE_RATIO"])
    social_count = total - personal_count

    for _ in range(personal_count):
        bees.append(
            Bee(
                strategy="personal",
                x=base_x + random.randint(-10, 10),
                y=base_y + random.randint(-10, 10),
                speed=cfg["BEE_SPEED"],
                q_values=[0.0] * len(flowers),
                epsilon=cfg["PERSONAL_EPSILON"],
                alpha=cfg["PERSONAL_ALPHA"],
                state="at_base",
            )
        )

    for _ in range(social_count):
        bees.append(
            Bee(
                strategy="social",
                x=base_x + random.randint(-10, 10),
                y=base_y + random.randint(-10, 10),
                speed=cfg["BEE_SPEED"],
                q_values=[0.0] * len(flowers),
                epsilon=cfg["PERSONAL_EPSILON"],
                alpha=cfg["PERSONAL_ALPHA"],
                state="at_base",
            )
        )

    zero_signal = [0] * len(flowers)
    for bee in bees:
        bee.choose_target(flowers, zero_signal, cfg)
        bee.state = "to_flower"

    return bees


class Simulation:
    def __init__(self, cfg, mode = "high"):
        self.cfg = cfg
        self.mode = mode
        self.frame_count = 0
        self.generation = 0
        self.step_counter = 0

        self.base_x = cfg["WORLD_WIDTH"] // 2
        self.base_y = cfg["HEIGHT"] // 2

        self.flowers = create_flowers(cfg, self.mode)
        self.bees = create_bees(cfg, self.flowers, self.base_x, self.base_y)

        self.last_counts = [0] * len(self.flowers)
        self.current_counts = [0] * len(self.flowers)
        self.social_visit_history = []

        self.last_social_fraction = self.social_fraction()
        self.last_avg_reward_personal = 0.0
        self.last_avg_reward_social = 0.0
        self.last_generation_social_count = 0
        self.last_generation_personal_count = 0
        self.last_generation_personal_mean_reward = 0.0
        self.last_generation_social_mean_reward = 0.0

        self.population_history_social = []
        self.population_history_personal = []
        self.reward_history_social = []
        self.reward_history_personal = []

    def social_fraction(self):
        social = sum(1 for b in self.bees if b.strategy == "social")
        return social / len(self.bees)

    def avg_rewards_total(self):
        personal = [b.reward_total for b in self.bees if b.strategy == "personal"]
        social = [b.reward_total for b in self.bees if b.strategy == "social"]

        avg_p = sum(personal) / max(1, len(personal))
        avg_s = sum(social) / max(1, len(social))
        return avg_p, avg_s

    def record_generation_stats(self):
        social_bees = [b for b in self.bees if b.strategy == "social"]
        personal_bees = [b for b in self.bees if b.strategy == "personal"]

        social_count = len(social_bees)
        personal_count = len(personal_bees)

        social_mean = sum(b.reward_total for b in social_bees) / max(1, social_count)
        personal_mean = sum(b.reward_total for b in personal_bees) / max(1, personal_count)

        self.last_generation_social_count = social_count
        self.last_generation_personal_count = personal_count
        self.last_generation_social_mean_reward = social_mean
        self.last_generation_personal_mean_reward = personal_mean

        self.population_history_social.append(social_count)
        self.population_history_personal.append(personal_count)
        self.reward_history_social.append(social_mean)
        self.reward_history_personal.append(personal_mean)

        trim_history(self.population_history_social, self.cfg["GRAPH_MAX_POINTS"])
        trim_history(self.population_history_personal, self.cfg["GRAPH_MAX_POINTS"])
        trim_history(self.reward_history_social, self.cfg["GRAPH_MAX_POINTS"])
        trim_history(self.reward_history_personal, self.cfg["GRAPH_MAX_POINTS"])

    def reset_world(self):
        self.flowers = create_flowers(self.cfg, self.mode)
        self.bees = create_bees(self.cfg, self.flowers, self.base_x, self.base_y)
        self.last_counts = [0] * len(self.flowers)
        self.current_counts = [0] * len(self.flowers)
        self.social_visit_history = []
        self.frame_count = 0
        self.generation = 0
        self.step_counter = 0

        self.population_history_social.clear()
        self.population_history_personal.clear()
        self.reward_history_social.clear()
        self.reward_history_personal.clear()

    def toggle_mode(self):
        self.mode = "stable" if self.mode == "high" else "high"
        self.reset_world()

    def reset_flower_stocks(self):
        for flower in self.flowers:
            flower.stock = flower.capacity

    def get_social_signal(self):
        if not self.social_visit_history:
            return [0] * len(self.flowers)

        aggregated = [0] * len(self.flowers)
        for counts in self.social_visit_history:
            for i in range(len(counts)):
                aggregated[i] += counts[i]
        return aggregated

    def step_environment(self):
        for flower in self.flowers:
            if self.cfg["ENABLE_DEPLETION"]:
                flower.stock += self.cfg["REGEN_RATE"] * (flower.capacity - flower.stock)

            if self.cfg["ENABLE_NOISE"]:
                flower.stock += random.gauss(0, self.cfg["NOISE_STD"])

            flower.stock = clamp(flower.stock, 0.0, flower.capacity)

    def step_bees(self):
        arrived_at_flower = []
        arrived_at_base = []

        for i, bee in enumerate(self.bees):
            if bee.state == "to_flower":
                if bee.target_flower_idx is None:
                    continue

                flower = self.flowers[bee.target_flower_idx]
                arrived = bee.move_towards_point(flower.x, flower.y)
                if arrived:
                    arrived_at_flower.append(i)

            elif bee.state == "to_base":
                arrived = bee.move_towards_point(self.base_x, self.base_y)
                if arrived:
                    arrived_at_base.append(i)

        self.current_counts = [0] * len(self.flowers)
        for i in arrived_at_flower:
            fidx = self.bees[i].target_flower_idx
            if fidx is not None:
                self.current_counts[fidx] += 1

        if self.cfg["ENABLE_DEPLETION"]:
            for flower_idx, count in enumerate(self.current_counts):
                self.flowers[flower_idx].stock = max(
                    0.0,
                    self.flowers[flower_idx].stock - self.cfg["HARVEST_STRENGTH"] * count
                )

        for i in arrived_at_flower:
            bee = self.bees[i]
            if bee.target_flower_idx is None:
                continue

            flower = self.flowers[bee.target_flower_idx]
            k = max(1, self.current_counts[bee.target_flower_idx])

            # Reward is reduced when many bees crowd the same flower
            nectar_reward = (
                (1.0 - self.cfg["CROWDING_PARAMETER"]) * flower.stock
                + self.cfg["CROWDING_PARAMETER"] * (flower.stock / k)
            )

            bee.trip_nectar_reward += nectar_reward
            bee.state = "to_base"

        self.social_visit_history.append(self.current_counts[:])
        if len(self.social_visit_history) > self.cfg["SOCIAL_MEMORY_LENGTH"]:
            self.social_visit_history.pop(0)

        social_signal = self.get_social_signal()

        for i in arrived_at_base:
            bee = self.bees[i]

            efficiency = (
                self.cfg["EFFICIENCY_SCALE"]
                * bee.trip_nectar_reward
                / max(bee.trip_distance, 1.0)
            )

            bee.reward_total += bee.trip_nectar_reward

            if bee.strategy == "personal" and bee.target_flower_idx is not None:
                bee.update_personal_memory(bee.target_flower_idx, efficiency)

            bee.trip_distance = 0.0
            bee.trip_nectar_reward = 0.0

            bee.choose_target(self.flowers, social_signal, self.cfg)
            bee.state = "to_flower"

        self.last_counts = self.current_counts[:]

    def evolve(self):
        # At the end of each generation, some bees are replaced
        # Parents are chosen with probability proportional to reward
        self.record_generation_stats()

        replace_count = min(self.cfg["BEES_REPLACED_PER_GENERATION"], len(self.bees))
        if replace_count <= 0:
            social_signal = self.get_social_signal()
            for bee in self.bees:
                bee.reset_for_new_generation(
                    self.base_x,
                    self.base_y,
                    self.flowers,
                    social_signal,
                    self.cfg
                )

            if self.cfg["RESET_FLOWERS_EACH_GENERATION"]:
                self.reset_flower_stocks()

            self.last_counts = [0] * len(self.flowers)
            self.current_counts = [0] * len(self.flowers)
            self.social_visit_history = []
            return

        fitness = [max(0.0, b.reward_total) for b in self.bees]
        total = sum(fitness)

        if total <= 0:
            probs = [1.0 / len(self.bees)] * len(self.bees)
        else:
            probs = [f / total for f in fitness]

        parents = random.choices(self.bees, weights = probs, k = replace_count)

        survivor_indices = list(range(len(self.bees)))
        dead_indices = set(random.sample(survivor_indices, replace_count))
        survivors = [bee for i, bee in enumerate(self.bees) if i not in dead_indices]

        newborns = []
        for parent in parents:
            strategy = parent.strategy

            if random.random() < self.cfg["MUTATION_RATE"]:
                strategy = "social" if strategy == "personal" else "personal"

            parent_q = parent.q_values[:] if self.cfg["INHERIT_PARENT_MEMORY"] else [0.0] * len(self.flowers)

            newborns.append(
                Bee(
                    strategy=strategy,
                    x=self.base_x + random.randint(-10, 10),
                    y=self.base_y + random.randint(-10, 10),
                    speed=self.cfg["BEE_SPEED"],
                    q_values=parent_q,
                    epsilon=self.cfg["PERSONAL_EPSILON"],
                    alpha=self.cfg["PERSONAL_ALPHA"],
                    state="at_base",
                )
            )

        self.bees = survivors + newborns

        if self.cfg["RESET_FLOWERS_EACH_GENERATION"]:
            self.reset_flower_stocks()

        self.last_counts = [0] * len(self.flowers)
        self.current_counts = [0] * len(self.flowers)
        self.social_visit_history = []

        social_signal = self.get_social_signal()
        for bee in self.bees:
            bee.reset_for_new_generation(
                self.base_x,
                self.base_y,
                self.flowers,
                social_signal,
                self.cfg
            )

    def update(self):
        self.frame_count += 1
        self.step_counter += 1

        self.step_environment()
        self.step_bees()

        if self.step_counter >= self.cfg["GENERATION_LENGTH"]:
            self.evolve()
            self.step_counter = 0
            self.generation += 1

        self.last_social_fraction = self.social_fraction()
        self.last_avg_reward_personal, self.last_avg_reward_social = self.avg_rewards_total()


def draw_text(screen, font, text, x, y, colour=(20, 20, 20)):
    surface = font.render(text, True, colour)
    screen.blit(surface, (x, y))


def draw_flower(screen, flower, cfg, font_small):
    frac = flower.stock_fraction()

    poor_empty = (220, 220, 220)
    poor_full = (120, 180, 255)
    rich_empty = (230, 220, 220)
    rich_full = (255, 170, 60)

    colour = colour_lerp(rich_empty, rich_full, frac) if flower.is_rich else colour_lerp(poor_empty, poor_full, frac)

    r = flower.radius(cfg)
    pygame.draw.circle(screen, colour, (int(flower.x), int(flower.y)), r)
    pygame.draw.circle(screen, (80, 80, 80), (int(flower.x), int(flower.y)), r, 2)

    if cfg["SHOW_DEBUG_RICH"] and flower.is_rich:
        pygame.draw.circle(screen, (255, 0, 0), (int(flower.x), int(flower.y)), r + 4, 2)

    if cfg["SHOW_FLOWER_LABELS"]:
        txt = f"{flower.stock:.1f} [{flower.idx}]" if cfg["SHOW_FLOWER_ID"] else f"{flower.stock:.1f}"
        draw_text(screen, font_small, txt, flower.x - 22, flower.y - r - 18, (50, 50, 50))


def draw_bee(screen, bee, flowers, cfg):
    colour = (40, 90, 220) if bee.strategy == "personal" else (220, 70, 70)
    pygame.draw.circle(screen, colour, (int(bee.x), int(bee.y)), cfg["BEE_RADIUS"])

    if cfg["SHOW_TARGET_LINES"] and bee.target_flower_idx is not None:
        if 0 <= bee.target_flower_idx < len(flowers):
            target = flowers[bee.target_flower_idx]
            pygame.draw.line(
                screen,
                colour,
                (int(bee.x), int(bee.y)),
                (int(target.x), int(target.y)),
                1
            )


def draw_base(screen, sim):
    pygame.draw.circle(screen, (60, 60, 60), (sim.base_x, sim.base_y), 18)
    pygame.draw.circle(screen, (180, 180, 180), (sim.base_x, sim.base_y), 12)


def draw_graph(screen, rect, series_a, series_b, colour_a, colour_b, title, label_a, label_b, font_small, bg=(252, 252, 252)):
    pygame.draw.rect(screen, bg, rect)
    pygame.draw.rect(screen, (180, 180, 180), rect, 2)

    draw_text(screen, font_small, title, rect.x + 8, rect.y + 6)

    inner_x = rect.x + 10
    inner_y = rect.y + 30
    inner_w = rect.width - 20
    inner_h = rect.height - 40

    if inner_w <= 0 or inner_h <= 0:
        return

    pygame.draw.line(screen, (210, 210, 210), (inner_x, inner_y + inner_h), (inner_x + inner_w, inner_y + inner_h), 1)
    pygame.draw.line(screen, (210, 210, 210), (inner_x, inner_y), (inner_x, inner_y + inner_h), 1)

    all_vals = list(series_a) + list(series_b)
    if not all_vals:
        draw_text(screen, font_small, "No data yet", inner_x + 10, inner_y + 10, (120, 120, 120))
        return

    ymin = min(all_vals)
    ymax = max(all_vals)
    if abs(ymax - ymin) < 1e-9:
        ymax = ymin + 1.0

    def to_points(series):
        if len(series) == 1:
            x = inner_x
            y = inner_y + inner_h - ((series[0] - ymin) / (ymax - ymin)) * inner_h
            return [(x, y)]

        pts = []
        for i, v in enumerate(series):
            px = inner_x + (i / (len(series) - 1)) * inner_w
            py = inner_y + inner_h - ((v - ymin) / (ymax - ymin)) * inner_h
            pts.append((px, py))
        return pts

    pts_a = to_points(series_a)
    pts_b = to_points(series_b)

    if len(pts_a) >= 2:
        pygame.draw.lines(screen, colour_a, False, pts_a, 2)
    elif len(pts_a) == 1:
        pygame.draw.circle(screen, colour_a, (int(pts_a[0][0]), int(pts_a[0][1])), 2)

    if len(pts_b) >= 2:
        pygame.draw.lines(screen, colour_b, False, pts_b, 2)
    elif len(pts_b) == 1:
        pygame.draw.circle(screen, colour_b, (int(pts_b[0][0]), int(pts_b[0][1])), 2)

    draw_text(screen, font_small, label_a, inner_x, rect.y + rect.height - 8, colour_a)
    draw_text(screen, font_small, label_b, inner_x + 120, rect.y + rect.height - 8, colour_b)
    draw_text(screen, font_small, f"min={ymin:.2f}", rect.x + rect.width - 120, rect.y + 6, (90, 90, 90))
    draw_text(screen, font_small, f"max={ymax:.2f}", rect.x + rect.width - 120, rect.y + 24, (90, 90, 90))


def draw_world_ui(screen, sim, font, font_small):
    cfg = sim.cfg
    x = 15
    y = 15
    line = 24

    draw_text(screen, font, "Bee Simulation: Personal VS Social Bees", x, y)
    y += line + 8

    rich_count = cfg["NUMBER_OF_FLOWERS"] if sim.mode == "stable" else cfg["RICH_COUNT_HIGH"]
    draw_text(screen, font_small, f"Mode: {sim.mode} | Rich: {rich_count}/{cfg['NUMBER_OF_FLOWERS']}", x, y)
    y += line
    draw_text(screen, font_small, f"Generation: {sim.generation} | Step in gen: {sim.step_counter}/{cfg['GENERATION_LENGTH']}", x, y)
    y += line
    draw_text(screen, font_small, f"Bees: {cfg['TOTAL_BEES']} | Replaced/gen: {cfg['BEES_REPLACED_PER_GENERATION']}", x, y)
    y += line
    draw_text(screen, font_small, f"Social fraction: {sim.last_social_fraction:.3f}", x, y)
    y += line
    draw_text(screen, font_small, f"Speed: {cfg['SIMULATION_SPEED']}x", x, y)
    y += line
    draw_text(screen, font_small, f"Efficiency scale: {cfg['EFFICIENCY_SCALE']:.1f}", x, y)
    y += line
    draw_text(screen, font_small, f"Softmax tau: {cfg['SOFTMAX_TAU']:.4f}", x, y)
    y += line
    draw_text(screen, font_small, f"Mutation: {cfg['MUTATION_RATE']:.3f}", x, y)
    y += line
    draw_text(screen, font_small, f"Social memory length: {cfg['SOCIAL_MEMORY_LENGTH']}", x, y)
    y += line
    draw_text(screen, font_small, f"Flower distance mode: {cfg['FLOWER_DISTANCE_MODE']}", x, y)
    y += line

    if cfg["FLOWER_DISTANCE_MODE"] == "constant":
        draw_text(screen, font_small, f"Constant distance: {cfg['FLOWER_DISTANCE_CONSTANT']}", x, y)
        y += line
    elif cfg["FLOWER_DISTANCE_MODE"] == "range":
        draw_text(screen, font_small, f"Distance range: {cfg['FLOWER_DISTANCE_MIN']} - {cfg['FLOWER_DISTANCE_MAX']}", x, y)
        y += line

    draw_text(screen, font_small, f"Regen: {cfg['REGEN_RATE']:.3f} | Harvest: {cfg['HARVEST_STRENGTH']:.3f}", x, y)
    y += line
    draw_text(screen, font_small, f"Noise: {cfg['ENABLE_NOISE']} | Noise std: {cfg['NOISE_STD']:.3f}", x, y)
    y += line
    draw_text(screen, font_small, f"Crowding: {cfg['CROWDING_PARAMETER']:.2f}", x, y)

    legend_y = cfg["HEIGHT"] - 70
    pygame.draw.circle(screen, (40, 90, 220), (20, legend_y), 6)
    draw_text(screen, font_small, "Personal", 35, legend_y - 10)
    pygame.draw.circle(screen, (220, 70, 70), (120, legend_y), 6)
    draw_text(screen, font_small, "Social", 135, legend_y - 10)
    pygame.draw.circle(screen, (60, 60, 60), (220, legend_y), 10)
    draw_text(screen, font_small, "Base", 240, legend_y - 10)


def draw_panel(screen, sim, font, font_small):
    cfg = sim.cfg
    panel_x = cfg["WORLD_WIDTH"]
    panel_rect = pygame.Rect(panel_x, 0, cfg["PANEL_WIDTH"], cfg["HEIGHT"])
    pygame.draw.rect(screen, (235, 235, 235), panel_rect)
    pygame.draw.line(screen, (180, 180, 180), (panel_x, 0), (panel_x, cfg["HEIGHT"]), 3)

    x = panel_x + 15
    y = 15
    line = 24

    draw_text(screen, font, "Analytics panel", x, y)
    y += line + 8

    draw_text(screen, font_small, f"Last gen personal count: {sim.last_generation_personal_count}", x, y)
    y += line
    draw_text(screen, font_small, f"Last gen social count: {sim.last_generation_social_count}", x, y)
    y += line
    draw_text(screen, font_small, f"Last gen personal mean reward: {sim.last_generation_personal_mean_reward:.2f}", x, y)
    y += line
    draw_text(screen, font_small, f"Last gen social mean reward: {sim.last_generation_social_mean_reward:.2f}", x, y)
    y += line
    draw_text(screen, font_small, "Controls: SPACE pause | R reset | M mode", x, y)
    y += line
    draw_text(screen, font_small, "1/2/3/4 speed | T lines | D rich | N noise", x, y)
    y += line
    draw_text(screen, font_small, "Up/Down harvest | Left/Right regen", x, y)


    graph_w = cfg["PANEL_WIDTH"] - 30
    graph_h = 280

    pop_rect = pygame.Rect(panel_x + 15, 250, graph_w, graph_h)
    rew_rect = pygame.Rect(panel_x + 15, 560, graph_w, graph_h)

    draw_graph(
        screen,
        pop_rect,
        sim.population_history_personal,
        sim.population_history_social,
        (40, 90, 220),
        (220, 70, 70),
        "Population over generations",
        "Personal",
        "Social",
        font_small,
    )

    draw_graph(
        screen,
        rew_rect,
        sim.reward_history_personal,
        sim.reward_history_social,
        (40, 90, 220),
        (220, 70, 70),
        "Average net reward per generation",
        "Personal",
        "Social",
        font_small,
    )


def main(cfg=None, start_mode=None):
    if cfg is None:
        cfg = copy.deepcopy(CFG)

    if start_mode is None:
        start_mode = "stable"

    random.seed(cfg["SEED"])

    pygame.init()
    screen = pygame.display.set_mode((cfg["WIDTH"], cfg["HEIGHT"]))
    pygame.display.set_caption("Bee simulation with social memory buffer")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 24)
    font_small = pygame.font.SysFont("arial", 18)

    sim = Simulation(cfg)
    paused = False
    running = True

    while running:
        clock.tick(cfg["FPS"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:
                    sim = Simulation(cfg, start_mode=start_mode)

                elif event.key == pygame.K_m:
                    sim.toggle_mode()

                elif event.key == pygame.K_t:
                    cfg["SHOW_TARGET_LINES"] = not cfg["SHOW_TARGET_LINES"]

                elif event.key == pygame.K_d:
                    cfg["SHOW_DEBUG_RICH"] = not cfg["SHOW_DEBUG_RICH"]

                elif event.key == pygame.K_n:
                    cfg["ENABLE_NOISE"] = not cfg["ENABLE_NOISE"]

                elif event.key == pygame.K_UP:
                    cfg["HARVEST_STRENGTH"] = min(2.0, cfg["HARVEST_STRENGTH"] + 0.05)

                elif event.key == pygame.K_DOWN:
                    cfg["HARVEST_STRENGTH"] = max(0.0, cfg["HARVEST_STRENGTH"] - 0.05)

                elif event.key == pygame.K_RIGHT:
                    cfg["REGEN_RATE"] = min(1.0, cfg["REGEN_RATE"] + 0.01)

                elif event.key == pygame.K_LEFT:
                    cfg["REGEN_RATE"] = max(0.0, cfg["REGEN_RATE"] - 0.01)

                elif event.key == pygame.K_RIGHTBRACKET:
                    cfg["NOISE_STD"] = min(5.0, cfg["NOISE_STD"] + 0.01)

                elif event.key == pygame.K_LEFTBRACKET:
                    cfg["NOISE_STD"] = max(0.0, cfg["NOISE_STD"] - 0.01)

                elif event.key == pygame.K_1:
                    cfg["SIMULATION_SPEED"] = 1

                elif event.key == pygame.K_2:
                    cfg["SIMULATION_SPEED"] = 5

                elif event.key == pygame.K_3:
                    cfg["SIMULATION_SPEED"] = 20

                elif event.key == pygame.K_4:
                    cfg["SIMULATION_SPEED"] = 100

        if not paused:
            for _ in range(cfg["SIMULATION_SPEED"]):
                sim.update()

        screen.fill(cfg["BACKGROUND"])

        world_rect = pygame.Rect(0, 0, cfg["WORLD_WIDTH"], cfg["HEIGHT"])
        pygame.draw.rect(screen, cfg["BACKGROUND"], world_rect)

        for flower in sim.flowers:
            draw_flower(screen, flower, cfg, font_small)

        draw_base(screen, sim)

        for bee in sim.bees:
            draw_bee(screen, bee, sim.flowers, cfg)

        draw_world_ui(screen, sim, font, font_small)
        draw_panel(screen, sim, font, font_small)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()