# FEs-Inverse-Design
This repository contains the source code and datasets for the research paper: "Active learning in latent spaces enables rapid inverse design of ferroelectric ceramics for energy storage".


---
![image](./workflow.png) <br>
---

## 🚀 Key Features
Conditional Variational Autoencoder (cVAE): Constructs a coupled search space that synergistically models chemical constraints and domain structure evolution.

1. Latent Space Optimization: Implements a two-stage multi-objective genetic algorithm (NSGA-II) to navigate the latent space for Pareto-optimal solutions.

2. Active Learning Surrogate: Uses symbolic regression and ensemble learning (CatBoost, XGBoost, etc.) to predict energy density and efficiency with uncertainty quantification.

3. Phase-Field Simulation: Generate reference domain structures for training.

## 📁Repository Structure

cVAE/: Code for training the Conditional Variational Autoencoder and reconstructing energy landscapes.

ActiveLearning/: Latent space optimization algorithm, Surrogate model as fitness function of optimization algorithm, uncertainty quantification.

README/: Docs

## Citation
Xi, Z., Wang, Z., Guo, C. et al. Active learning in latent spaces enables rapid inverse design of ferroelectric ceramics for energy storage. **_Nat Commun_** (2026). https://doi.org/10.1038/s41467-026-70792-7
