# Bee Foraging Simulation - Social vs Personal Learning

This project implements an agent-based simulation of bee foraging behaviour, comparing social learning (copying others) and personal learning (reinforcement-based individual learning).

The model is based on:

Smolla et al. (2016)<br>
Copy-when-uncertain: bumblebees rely on social information when rewards are highly variable<br>
Biology Letters, 12: 20160188<br>
DOI: https://doi.org/10.1098/rsbl.2016.0188<br>
## Project Overview<br>

The original paper investigates how environmental reward variability affects the success of social and personal learning strategies in bumblebees.

This project reproduces that idea and extends it by adding:

- spatial flower environments
- travel costs
- resource depletion and regeneration
- environmental noise
- dynamic environments
- interactive visualisation using Pygame
- batch experiments with CSV output and graphs

## Files
bee_simulation.py - Main interactive simulation with full visualisation.

launcher.py - Preset launcher for quickly running different simulation worlds.

batch_runner.py - Runs repeated simulations and records population changes over generations.

variance_batch_runner.py - Runs reward variance experiments and generates final comparison graphs.

version1_non_visual_bee_simulation.py - Earlier non-visual prototype version.

sample_results/ - Example output graphs and CSV files.

## How to Run
Clone Repository
```
git clone https://github.com/N1kTop/Bee-Foraging-Simualtion-Personal-VS-Social-Leraners
```
Install requirements
```
pip install -r requirements.txt
```
Run interactive simulation
```
python bee_simulation.py
```
Or run preset launcher
```
python launcher.py
```

This allows selection of predefined environments such as:
- stable world
- high-variance world
- dynamic environments
- presentation/demo presets

Or run batch experiments
```
python batch_runner.py
```

Or run variance batch experiment
```
python variance_batch_runner.py
```

These generate CSV results and plots automatically.

## Controls

SPACE = pause / resume<br>
R = reset simulation<br>
M = switch mode (stable / high)<br>
1 / 2 / 3 / 4 = simulation speed<br>
T = show target lines<br>
D = show rich flowers<br>
N = toggle environmental noise

## Key Findings

Results matched the original paper:
- Stable environments favour personal learning
- High-variance environments favour social learning

Adding dynamic environmental effects balances the ratio of bee types.

Increasing personal learning in High-variance and social learning in Stable environments.

This suggests that environmental instability plays a major role in strategy selection.
