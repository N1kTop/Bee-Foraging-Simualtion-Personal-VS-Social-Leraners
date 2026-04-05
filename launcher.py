import copy
import random
import pygame

from bee_simulation import CFG, Simulation, draw_flower, draw_bee, draw_base, draw_world_ui, draw_panel


PRESETS = {
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


def main(preset_name="paper_high_variance"):
    cfg, start_mode = build_cfg(preset_name)

    random.seed(cfg["SEED"])

    pygame.init()
    screen = pygame.display.set_mode((cfg["WIDTH"], cfg["HEIGHT"]))
    pygame.display.set_caption(f"Bee simulation - {preset_name}")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("arial", 24)
    font_small = pygame.font.SysFont("arial", 18)

    sim = Simulation(cfg)
    sim.mode = start_mode
    sim.reset_world()

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
                    sim = Simulation(cfg)
                    sim.mode = start_mode
                    sim.reset_world()

        if not paused:
            for _ in range(cfg["SIMULATION_SPEED"]):
                sim.update()

        screen.fill(cfg["BACKGROUND"])

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
    print("Available presets:")
    for name in PRESETS:
        print("-", name)

    choice = input("Enter preset name: ").strip()
    main(choice)