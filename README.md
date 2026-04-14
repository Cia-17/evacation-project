# Sikaville Stadium — Evacuation Risk 

**Group 5**

---

## Overview

This project is an agent-based simulation model built to assess the emergency evacuation performance of Sikaville Stadium (5,000-seat capacity). It models how crowds behave under three distinct emergency scenarios — normal, partial blockage, and full panic — and produces quantitative metrics and safety recommendations to support the stadium's final safety certification.

Two components make up the system: a real-time Pygame visual simulation and a batch analysis pipeline that generates charts and an HTML report.

---

## Files

| File | Purpose |
|---|---|
| `evacuation_sims2.py` | Real-time Pygame animation of the evacuation |
| `analysisplot2.py` | Batch simulation + statistical analysis + HTML report |
| `results/` | Auto-generated output folder (created on first run) |

---

## Requirements

Install dependencies before running:

```bash
pip install pygame numpy pandas plotly scipy
```

Python 3.9 or higher is recommended.

---

## How to Run

### 1. Real-Time Visual Simulation (`evacuation_sims2.py`)

```bash
python evacuation_sims2.py
```

An interactive Pygame window opens showing the stadium layout with 5,000 agents.

**Controls:**

| Key / Click | Action |
|---|---|
| Click scenario name | Switch between Normal / Partial Block / Panic |
| `RUN` button or `SPACE` | Start the simulation |
| `RESET` button or `R` | Reset to initial state |
| `ESC` | Quit |

The right-side panel shows live stats: people evacuated, percentage done, elapsed time, milestone timestamps, and per-exit queue depths. Agents are colour-coded by proximity to their target exit (green → red as they get closer and more congested).

---

### 2. Batch Analysis & Report (`analysisplot2.py`)

```bash
python analysisplot2.py
```

Runs all three scenarios silently (no window), then writes the following to the `results/` folder:

| Output file | Contents |
|---|---|
| `report2.html` | Full HTML report with all charts embedded |
| `plot1_timeline.html` | Cumulative evacuation curves + logistic model fit |
| `plot2_milestones.html` | Time-to-clear bar chart (25 / 50 / 75 / 90 / 100%) |
| `plot3_bottleneck.html` | Max queue depth and average wait time per exit |
| `plot4_density.html` | Peak crowd density (people/m²) per stadium gate |
| `plot5_flowrate.html` | Rolling evacuation flow rate (people per second) |
| `timeline_*.csv` | Raw per-second evacuation counts for each scenario |
| `milestones.csv` | Milestone timestamps table |

All the csv files were initially generated with the intention of using them as hard-coded values for pygame but was not implemented.

Open `results/report2.html` in any browser to view the complete report.

---

## Scenarios

| Scenario | Exits Blocked | Speed Multiplier | Panic Factor |
|---|---|---|---|
| **Normal** | None | 1.0× | 1.0 |
| **Partial Block** | NE, SW | 0.80× | 1.3 |
| **Panic** | NE, SW | 0.60× | 2.0 |

All scenarios simulate 5,000 agents placed randomly in the oval seating bowl. Each agent moves toward their nearest open exit, with speed reduced by local crowd density and the scenario's panic factor.

---

## Simulation Model

### Agent Behaviour
- Each person is assigned the nearest open exit at initialisation.
- Movement speed is scaled by distance to exit (slower in queues, faster when clear) and a density-based slowdown factor.
- Agents apply a mild repulsion force to neighbours within 10 pixels to simulate personal space.
- An agent is considered evacuated when they reach within 6 pixels of their target exit.

### Stadium Layout
- **Centre:** (350, 290) in pixel space
- **8 exits** placed radially: N, NE, E, SE, S, SW, W, NW
- **Scale:** ~6.67 pixels per metre (based on a 105 m standard pitch width across 700 px)

### Metrics Tracked
- **Milestones:** Time (seconds) to evacuate 25 / 50 / 75 / 90 / 100% of occupants
- **Queue depth:** Maximum number of agents within 40 px of each exit at any point
- **Average wait time:** Mean time agents spend queuing at each exit
- **Peak crowd density:** Maximum agents/m² measured within a defined zone radius around each exit
- **Flow rate:** Rolling 5-second average of agents evacuating per second
- **Logistic fit:** Curve-fitted K (capacity), r (rate), and t₀ (midpoint) parameters for each scenario

---

## Safety Recommendations

The analysis identifies several improvements ranked by safety impact and implementation cost:

1. **Add 2 east-side exits** — highest safety impact; addresses the eastward bottleneck under partial-block scenarios
2. **Widen exits E and W** — reduces queue depths significantly at moderate cost
3. **Keep exits clear (5 m exclusion zone)** — zero-cost policy, immediate implementation
4. **Dynamic exit signs** — tech intervention to redistribute crowd load in real time
5. **Train exit stewards** — low cost, measurable reduction in panic factor
6. **Run evacuation drills** — reduces evacuation time by familiarising occupants with routes
7. **Reduce capacity to 4,200** — eliminates critical density thresholds; trade-off with revenue
8. **Upgrade emergency lighting** — supports all scenarios, especially panic conditions

See `plot6_recommendations.html` for the full cost vs. impact bubble chart.

---

## Key Findings

- Under **Normal** conditions, 90% of occupants evacuate well within the 180-second safety target.
- **Partial Block** (2 exits closed) pushes the 90% milestone past the 180 s threshold, creating critical bottlenecks at the remaining nearby exits.
- **Panic** conditions with the same blocked exits result in the longest evacuation times and the highest peak crowd densities, with several gates exceeding the critical 7 people/m² threshold.
- The logistic model fits all three scenarios well, with the panic scenario showing a significantly lower rate constant *r* and a later midpoint *t₀*.

---

## Notes

- The random seed is fixed (`np.random.default_rng(42)`) for reproducibility across all runs.
- The batch simulation (`analysisplot2.py`) uses a vectorised NumPy loop and runs at 60 simulated frames per second internally, making it significantly faster than the real-time Pygame version.
- Density calculations use a zone radius of 8 pixels (~1.2 m) around each exit, converted to m² for the people/m² metric.

---
