import math
import random
import pygame
from dataclasses import dataclass, field


# ============================================================
# CONFIG
# ============================================================

CFG = {
    # Window
    "WIDTH": 1400,
    "HEIGHT": 900,
    "FPS": 60,
    "BACKGROUND": (245, 245, 245),

    # World
    "WORLD_MARGIN": 80,
    "NUMBER_OF_FLOWERS": 24,
    "RICH_COUNT": 4,
    "MEAN_REWARD_PER_FLOWER": 10.0,
    "RICH_VS_POOR_REWARD_RATIO": 20.0,

    # Bee population
    "TOTAL_BEES": 80,
    "STARTING_BEE_RATIO": 0.5,   # personal ratio
    "BEE_SPEED": 1.6,
    "BEE_RADIUS": 4,

    # Bee strategies
    "PERSONAL_EPSILON": 0.10,
    "PERSONAL_ALPHA": 0.20,

    # Social strategy
    "SOCIAL_OBSERVATION_NOISE": 0.0,  # 0 = exact occupancy weighting

    # Dynamic environment
    "ENABLE_DEPLETION": True,
    "HARVEST_STRENGTH": 0.25,
    "REGEN_RATE": 0.03,
    "ENABLE_NOISE": False,
    "NOISE_STD": 0.02,

    # Reward crowding
    "CROWDING_PARAMETER": 0.4,  # 0 = no sharing, 1 = equal sharing

    # Visuals
    "FLOWER_MIN_RADIUS": 10,
    "FLOWER_MAX_RADIUS": 22,
    "SHOW_TARGET_LINES": False,
    "SHOW_FLOWER_LABELS": True,
    "SHOW_DEBUG_RICH": False,

    # Timing
    "SIMULATION_STEPS_PER_FRAME": 1,
    "AUTO_RESPAWN_ENVIRONMENT": False,
    "RESPAWN_INTERVAL_FRAMES": 0,

    # Seed
    "SEED": 42,
}


# ============================================================
# HELPERS
# ============================================================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def weighted_choice(indices, weights):
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


# ============================================================
# FLOWERS
# ============================================================

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


# ============================================================
# BEES
# ============================================================

@dataclass
class Bee:
    strategy: str  # "personal" or "social"
    x: float
    y: float
    speed: float
    reward_total: float = 0.0
    target_flower_idx: int | None = None
    q_values: list[float] = field(default_factory=list)
    epsilon: float = 0.1
    alpha: float = 0.2

    def choose_target(self, flowers, last_counts):
        if self.strategy == "personal":
            if random.random() < self.epsilon:
                self.target_flower_idx = random.randrange(len(flowers))
                return

            max_q = max(self.q_values)
            best = [i for i, q in enumerate(self.q_values) if q == max_q]
            self.target_flower_idx = random.choice(best)
            return

        indices = list(range(len(flowers)))
        noisy_counts = []
        for c in last_counts:
            noisy = max(0.0, c + random.gauss(0, CFG["SOCIAL_OBSERVATION_NOISE"]))
            noisy_counts.append(noisy)

        if sum(noisy_counts) <= 0:
            self.target_flower_idx = random.randrange(len(flowers))
        else:
            self.target_flower_idx = weighted_choice(indices, noisy_counts)

    def move_towards_target(self, flowers):
        if self.target_flower_idx is None:
            return True

        flower = flowers[self.target_flower_idx]
        dx = flower.x - self.x
        dy = flower.y - self.y
        d = math.hypot(dx, dy)

        if d <= self.speed or d == 0:
            self.x = flower.x
            self.y = flower.y
            return True

        self.x += self.speed * dx / d
        self.y += self.speed * dy / d
        return False

    def update_personal_memory(self, flower_idx, reward):
        old = self.q_values[flower_idx]
        self.q_values[flower_idx] = old + self.alpha * (reward - old)


# ============================================================
# WORLD SETUP
# ============================================================

def generate_capacities(cfg):
    n = cfg["NUMBER_OF_FLOWERS"]
    rich_count = cfg["RICH_COUNT"]
    total_reward = n * cfg["MEAN_REWARD_PER_FLOWER"]
    ratio = cfg["RICH_VS_POOR_REWARD_RATIO"]

    denominator = rich_count + (n - rich_count) / ratio
    rich_reward = total_reward / denominator
    poor_reward = rich_reward / ratio

    caps = [rich_reward] * rich_count + [poor_reward] * (n - rich_count)
    random.shuffle(caps)
    cfg["_all_capacities"] = caps[:]
    return caps


def create_flowers(cfg):
    caps = generate_capacities(cfg)
    flowers = []

    w = cfg["WIDTH"]
    h = cfg["HEIGHT"]
    m = cfg["WORLD_MARGIN"]

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
                is_rich=(cap == max(caps)),
            )
        )
    return flowers


def create_bees(cfg, flowers):
    bees = []
    total = cfg["TOTAL_BEES"]
    personal_count = int(total * cfg["STARTING_BEE_RATIO"])
    social_count = total - personal_count

    centre_x = cfg["WIDTH"] // 2
    centre_y = cfg["HEIGHT"] // 2

    for _ in range(personal_count):
        bees.append(
            Bee(
                strategy="personal",
                x=centre_x + random.randint(-40, 40),
                y=centre_y + random.randint(-40, 40),
                speed=cfg["BEE_SPEED"],
                q_values=[0.0] * len(flowers),
                epsilon=cfg["PERSONAL_EPSILON"],
                alpha=cfg["PERSONAL_ALPHA"],
            )
        )

    for _ in range(social_count):
        bees.append(
            Bee(
                strategy="social",
                x=centre_x + random.randint(-40, 40),
                y=centre_y + random.randint(-40, 40),
                speed=cfg["BEE_SPEED"],
                q_values=[0.0] * len(flowers),
                epsilon=cfg["PERSONAL_EPSILON"],
                alpha=cfg["PERSONAL_ALPHA"],
            )
        )

    for bee in bees:
        bee.choose_target(flowers, [0] * len(flowers))

    return bees


# ============================================================
# SIMULATION
# ============================================================

class Simulation:
    def __init__(self, cfg):
        self.cfg = cfg
        self.frame_count = 0
        self.generation = 0
        self.step_count = 0

        self.flowers = create_flowers(cfg)
        self.bees = create_bees(cfg, self.flowers)

        self.last_counts = [0] * len(self.flowers)
        self.current_counts = [0] * len(self.flowers)

        self.last_social_fraction = self.social_fraction()
        self.last_avg_reward_personal = 0.0
        self.last_avg_reward_social = 0.0

    def social_fraction(self):
        social = sum(1 for b in self.bees if b.strategy == "social")
        return social / len(self.bees)

    def avg_rewards(self):
        personal = [b.reward_total for b in self.bees if b.strategy == "personal"]
        social = [b.reward_total for b in self.bees if b.strategy == "social"]

        avg_p = sum(personal) / max(1, len(personal))
        avg_s = sum(social) / max(1, len(social))
        return avg_p, avg_s

    def maybe_respawn_environment(self):
        if self.cfg["AUTO_RESPAWN_ENVIRONMENT"]:
            interval = self.cfg["RESPAWN_INTERVAL_FRAMES"]
            if interval > 0 and self.frame_count % interval == 0 and self.frame_count > 0:
                self.flowers = create_flowers(self.cfg)
                self.last_counts = [0] * len(self.flowers)
                self.current_counts = [0] * len(self.flowers)
                for bee in self.bees:
                    bee.q_values = [0.0] * len(self.flowers)
                    bee.target_flower_idx = None
                    bee.choose_target(self.flowers, self.last_counts)

    def step_environment(self):
        for flower in self.flowers:
            if self.cfg["ENABLE_DEPLETION"]:
                flower.stock += self.cfg["REGEN_RATE"] * (flower.capacity - flower.stock)

            if self.cfg["ENABLE_NOISE"]:
                flower.stock += random.gauss(0, self.cfg["NOISE_STD"])

            flower.stock = clamp(flower.stock, 0.0, flower.capacity)

    def step_bees(self):
        arrived_indices = []

        for i, bee in enumerate(self.bees):
            arrived = bee.move_towards_target(self.flowers)
            if arrived and bee.target_flower_idx is not None:
                arrived_indices.append(i)

        self.current_counts = [0] * len(self.flowers)
        for i in arrived_indices:
            fidx = self.bees[i].target_flower_idx
            self.current_counts[fidx] += 1

        if self.cfg["ENABLE_DEPLETION"]:
            for flower_idx, count in enumerate(self.current_counts):
                self.flowers[flower_idx].stock = max(
                    0.0,
                    self.flowers[flower_idx].stock - self.cfg["HARVEST_STRENGTH"] * count
                )

        for i in arrived_indices:
            bee = self.bees[i]
            flower = self.flowers[bee.target_flower_idx]
            k = max(1, self.current_counts[bee.target_flower_idx])

            reward = (1.0 - self.cfg["CROWDING_PARAMETER"]) * flower.stock + \
                     self.cfg["CROWDING_PARAMETER"] * (flower.stock / k)

            bee.reward_total += reward

            if bee.strategy == "personal":
                bee.update_personal_memory(bee.target_flower_idx, reward)

        for i in arrived_indices:
            self.bees[i].choose_target(self.flowers, self.current_counts)

        self.last_counts = self.current_counts[:]

    def update(self):
        self.frame_count += 1
        self.step_count += 1

        self.maybe_respawn_environment()
        self.step_environment()
        self.step_bees()

        self.last_social_fraction = self.social_fraction()
        self.last_avg_reward_personal, self.last_avg_reward_social = self.avg_rewards()


# ============================================================
# DRAWING
# ============================================================

def draw_text(screen, font, text, x, y, colour=(20, 20, 20)):
    surface = font.render(text, True, colour)
    screen.blit(surface, (x, y))


def draw_flower(screen, flower, cfg, font_small):
    frac = flower.stock_fraction()

    poor_empty = (220, 220, 220)
    poor_full = (120, 180, 255)
    rich_empty = (230, 220, 220)
    rich_full = (255, 170, 60)

    if flower.is_rich:
        colour = colour_lerp(rich_empty, rich_full, frac)
    else:
        colour = colour_lerp(poor_empty, poor_full, frac)

    r = flower.radius(cfg)
    pygame.draw.circle(screen, colour, (int(flower.x), int(flower.y)), r)
    pygame.draw.circle(screen, (80, 80, 80), (int(flower.x), int(flower.y)), r, 2)

    if cfg["SHOW_DEBUG_RICH"] and flower.is_rich:
        pygame.draw.circle(screen, (255, 0, 0), (int(flower.x), int(flower.y)), r + 4, 2)

    if cfg["SHOW_FLOWER_LABELS"]:
        txt = f"{flower.idx}:{flower.stock:.1f}"
        draw_text(screen, font_small, txt, flower.x - 18, flower.y - r - 18, (50, 50, 50))


def draw_bee(screen, bee, flowers, cfg):
    if bee.strategy == "personal":
        colour = (40, 90, 220)
    else:
        colour = (220, 70, 70)

    pygame.draw.circle(screen, colour, (int(bee.x), int(bee.y)), cfg["BEE_RADIUS"])

    if cfg["SHOW_TARGET_LINES"] and bee.target_flower_idx is not None:
        flower = flowers[bee.target_flower_idx]
        pygame.draw.line(
            screen,
            colour,
            (int(bee.x), int(bee.y)),
            (int(flower.x), int(flower.y)),
            1
        )


def draw_ui(screen, sim, font, font_small):
    cfg = sim.cfg
    x = 15
    y = 15
    line = 24

    draw_text(screen, font, "Ver6 top-down bee simulation", x, y)
    y += line + 8

    draw_text(screen, font_small, f"Flowers: {cfg['NUMBER_OF_FLOWERS']} | Rich: {cfg['RICH_COUNT']}", x, y)
    y += line
    draw_text(screen, font_small, f"Bees: {cfg['TOTAL_BEES']} | Personal ratio: {cfg['STARTING_BEE_RATIO']:.2f}", x, y)
    y += line
    draw_text(screen, font_small, f"Social fraction: {sim.last_social_fraction:.3f}", x, y)
    y += line
    draw_text(screen, font_small, f"Avg personal reward: {sim.last_avg_reward_personal:.2f}", x, y)
    y += line
    draw_text(screen, font_small, f"Avg social reward: {sim.last_avg_reward_social:.2f}", x, y)
    y += line
    draw_text(screen, font_small, f"Regen: {cfg['REGEN_RATE']:.3f} | Harvest: {cfg['HARVEST_STRENGTH']:.3f}", x, y)
    y += line
    draw_text(screen, font_small, f"Noise on: {cfg['ENABLE_NOISE']} | Noise std: {cfg['NOISE_STD']:.3f}", x, y)
    y += line
    draw_text(screen, font_small, f"Crowding: {cfg['CROWDING_PARAMETER']:.2f}", x, y)
    y += line
    draw_text(screen, font_small, "Controls: SPACE pause | R reset | T target lines | D rich debug | N noise toggle", x, y)
    y += line
    draw_text(screen, font_small, "          Up/Down harvest | Left/Right regen | [ / ] noise std", x, y)

    legend_y = cfg["HEIGHT"] - 70
    pygame.draw.circle(screen, (40, 90, 220), (20, legend_y), 6)
    draw_text(screen, font_small, "Personal", 35, legend_y - 10)
    pygame.draw.circle(screen, (220, 70, 70), (120, legend_y), 6)
    draw_text(screen, font_small, "Social", 135, legend_y - 10)


# ============================================================
# MAIN LOOP
# ============================================================

def main():
    random.seed(CFG["SEED"])

    pygame.init()
    screen = pygame.display.set_mode((CFG["WIDTH"], CFG["HEIGHT"]))
    pygame.display.set_caption("Bee top-down simulation")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 24)
    font_small = pygame.font.SysFont("arial", 18)

    sim = Simulation(CFG)
    paused = False
    running = True

    while running:
        dt = clock.tick(CFG["FPS"])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

                elif event.key == pygame.K_r:
                    sim = Simulation(CFG)

                elif event.key == pygame.K_t:
                    CFG["SHOW_TARGET_LINES"] = not CFG["SHOW_TARGET_LINES"]

                elif event.key == pygame.K_d:
                    CFG["SHOW_DEBUG_RICH"] = not CFG["SHOW_DEBUG_RICH"]

                elif event.key == pygame.K_n:
                    CFG["ENABLE_NOISE"] = not CFG["ENABLE_NOISE"]

                elif event.key == pygame.K_UP:
                    CFG["HARVEST_STRENGTH"] = min(2.0, CFG["HARVEST_STRENGTH"] + 0.05)

                elif event.key == pygame.K_DOWN:
                    CFG["HARVEST_STRENGTH"] = max(0.0, CFG["HARVEST_STRENGTH"] - 0.05)

                elif event.key == pygame.K_RIGHT:
                    CFG["REGEN_RATE"] = min(1.0, CFG["REGEN_RATE"] + 0.01)

                elif event.key == pygame.K_LEFT:
                    CFG["REGEN_RATE"] = max(0.0, CFG["REGEN_RATE"] - 0.01)

                elif event.key == pygame.K_RIGHTBRACKET:
                    CFG["NOISE_STD"] = min(5.0, CFG["NOISE_STD"] + 0.01)

                elif event.key == pygame.K_LEFTBRACKET:
                    CFG["NOISE_STD"] = max(0.0, CFG["NOISE_STD"] - 0.01)

        if not paused:
            for _ in range(CFG["SIMULATION_STEPS_PER_FRAME"]):
                sim.update()

        screen.fill(CFG["BACKGROUND"])

        for flower in sim.flowers:
            draw_flower(screen, flower, CFG, font_small)

        for bee in sim.bees:
            draw_bee(screen, bee, sim.flowers, CFG)

        draw_ui(screen, sim, font, font_small)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()