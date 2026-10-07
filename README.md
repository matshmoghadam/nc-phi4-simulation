# Noncommutative $\phi^4$ Field Theory Simulation

Numerical simulation and spectral study of 2D classical and noncommutative scalar field dynamics using the Moyal star product.

## 📌 Overview
This project simulates the collision of Gaussian wave packets in a $\phi^4$ scalar field theory on a noncommutative plane. It compares classical continuum field dynamics with noncommutative field theory (NCFT) evaluated via spectral methods.

## 📁 Repository Structure
```text
nc-phi4-simulation/
├── assets/                  # Figures and animation output
│   ├── static_deformation.png
│   ├── metrics_plot.png
│   ├── frame_01.png
│   ├── frame_02.png
│   ├── frame_03.png
│   ├── frame_04.png
│   └── collision_animation.gif
│   └── collision_animation.mp4
├── src/                     # Python simulation scripts
│   ├── static_moyal_check.py
│   ├── simulate_collision.py
│   └── analyze_metrics.py
├── .gitignore
├── paper.pdf                # Full LaTeX compiled paper
├── README.md
└── requirements.txt