# FEs-Inverse-Design
This repository contains the source code and datasets for the research paper: "Active learning in latent spaces accelerates inverse design of ferroelectric ceramics for energy storage".


---
![image](./workflow.png) <br>
---

🚀 Key Features
Conditional Variational Autoencoder (cVAE): Constructs a coupled search space that synergistically models chemical constraints and domain structure evolution.

1. Latent Space Optimization: Implements a two-stage multi-objective genetic algorithm (NSGA-II) to navigate the latent space for Pareto-optimal solutions.

2. Active Learning Surrogate: Uses symbolic regression and ensemble learning (CatBoost, XGBoost, etc.) to predict energy density and efficiency with uncertainty quantification.

3. Phase-Field Simulation: Generate reference domain structures for training.

📁Repository Structure

cVAE/: Code for training the Conditional Variational Autoencoder and reconstructing energy landscapes.

ActiveLearning/: Surrogate model training, uncertainty quantification.
