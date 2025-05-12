# Schrödinger Equation Solver in 1D and 2D

**Author:** Moulik Mishra
**Project Type:** Computational Quantum Mechanics  
**Programming Language:** Python  
**Status:** Completed

---

## Overview

This project implements a high-precision numerical solution to the **time-independent Schrödinger equation (TISE)** using **finite difference methods** in both 1D and 2D spatial domains. The solver computes energy eigenvalues and eigenstates for a wide range of quantum potentials, and provides clear visualizations of wavefunctions, probability densities, and quantum energy levels.

The project is fully modular, efficient (using sparse matrices and SciPy's eigensolvers), and designed to be easily extensible for more complex or custom quantum systems.

---

## Key Features

- Solves the time-independent Schrödinger equation in both 1D and 2D
- Uses finite difference discretization and sparse matrix techniques
- Handles multiple potential types including:
  - Infinite square well
  - Harmonic oscillator
  - Finite well
  - Double well
  - Morse potential
  - Kronig-Penney (1D periodic)
  - Coupled 2D oscillators
  - Crater and Gaussian wells in 2D
- Computes and normalizes eigenstates
- Plots:
  - Individual wavefunctions and energy levels
  - Probability densities
  - 2D contour, density, and 3D surface plots
- Saves all results (eigenvalues, eigenvectors, plots) to disk

---

## Project Structure

- `schrodinger-solver.py` – Main solver script (all classes and execution)
- `results/` – Directory created automatically to store:
  - `.dat` files for eigenvalues, eigenvectors, and potentials
  - `.png` visualizations of all states and levels

---

## How to Run

1. Clone the repository or copy the script locally.
2. Install the required dependencies:

   ```bash
   pip install numpy scipy matplotlib

3. The program will:

- Generate synthetic galaxy images  
- Train a CNN model  
- Evaluate its performance  
- Save training plots and Grad-CAM visualizations  

## Outputs

- `training_history.png`: Accuracy and loss across training epochs  
- `confusion_matrix.png`: Confusion matrix of predictions  
- `gradcam_overview.png`: Grid of Grad-CAM visualizations by class  
- `results/gradcam/`: Folder of individual Grad-CAM images  
- Trained model saved as `models/galaxy_cnn_best.h5`

---

## Scientific Relevance

Galaxy morphology is closely linked to formation history and dynamical processes.  
Automating galaxy classification is critical for large surveys.  
This project demonstrates how custom deep learning models can simulate and classify galaxies in a scientifically interpretable and scalable way.

---

## Future Work

- Extend the model to classify real images from Galaxy Zoo.  
- Introduce more complex or hybrid architectures (e.g., residual or attention-based CNNs).  
- Deploy the classifier as a simple web tool with interactive Grad-CAM outputs.

---

## Contact

**Email**: [mm748@snu.edu.in](mailto:mm748@snu.edu.in)  
**Affiliation**: Department of Physics, Shiv Nadar University

