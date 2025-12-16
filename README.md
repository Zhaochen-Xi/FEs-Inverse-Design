# FEs-Inverse-Design
This repository contains the source code and datasets for the research paper: "Active learning optimization in latent spaces accelerates inverse design of ferroelectric ceramics for energy storage".

📝 Overview
Simultaneously achieving high energy density and efficiency in ferroelectric ceramics is challenging due to the complex coupling between chemical composition and polarization configuration. Traditional high-throughput exploration is often limited by the computational cost of solving real-time polarization dynamics.This project implements an inverse design framework that integrates a variational generative model with active learning-driven optimization. By formulating the time-dependent Ginzburg-Landau (TDGL) equation as conditional sampling within a latent space, we accelerate the discovery of BNT-based relaxor-ferroelectrics with superior energy storage performance.

---
![image](./worflow.png) <br>
---

🚀 Key Features
Conditional Variational Autoencoder (cVAE): Constructs a coupled search space that synergistically models chemical constraints and domain structure evolution.

1. Latent Space Optimization: Implements a two-stage multi-objective genetic algorithm (NSGA-II) to navigate the latent space for Pareto-optimal solutions.

2. Active Learning Surrogate: Uses symbolic regression and ensemble learning (CatBoost, XGBoost, etc.) to predict energy density and efficiency with uncertainty quantification.

3. Phase-Field Simulation: Generate reference domain structures for training.

📁Repository Structure

cVAE/: Code for training the Conditional Variational Autoencoder and reconstructing energy landscapes.

ActiveLearning/: Surrogate model training, uncertainty quantification.
