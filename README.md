# NN-HiSD

NN-HiSD, a method which utilizes a neural network surrogate model to approximate the energy function and employs HiSD on this surrogate model to compute saddle points.

## File Directory Structure

```
├── core/                  # Core algorithm implementation
│   ├── hisd.py            # HiSD algorithm core implementation
│   ├── hisdnn.py          # Neural network-based HiSD
│   ├── trainer.py         # Neural network trainer
│   └── __init__.py
├── examples/              # Example folders
│   ├── ex1_simple/        # Simple potential model examples
│   │   └── ...            # configurations
│   ├── ex2_mb/            # Muller-Brown potential examples
│   │   └── ...
│   ├── ex3_rosenbrock/    # Rosenbrock potential examples
│   │   └── ...
│   ├── ex4_nanma/         # Alanine dipeptide model examples
│   │   └── ...
│   └── ex5_bacterial/     # Bacterial model examples   
│       └── ...
├── utils/                 # Utility functions
│   ├── createdata.py      # Dataset creation tools
│   ├── hessian_utils.py   # Hessian matrix calculation tools
│   ├── networks.py        # Neural network structure 
|   └── __init__.py
├── train.py               # Main neural network training script
├── requirements.txt       # Environment dependencies
└── README.md              # Project documentation
```

### Directory Structure Description

- **core/**: Contains the core implementation of the algorithm, including the basic HiSD algorithm implementation, neural network-based extension, and neural network trainer.

- **examples/**: Contains 5 different examples, ranging from simple potential models to complex systems:
  - **ex1_simple/**: Simple 2D and 3D potential models for demonstrating the basic working principles of the algorithm.
  - **ex2_mb/**: Muller-Brown potential model examples comparing the effects with and without gradient information.
  - **ex3_rosenbrock/**: Rosenbrock potential model examples studying the impact of different point numbers on results.
  - **ex4_nanma/**: Alanine dipeptide molecular model demonstrating the algorithm's application in biomolecular systems.
  - **ex5_bacterial/**: Bacterial model examples building surrogate models based on experimental data.

- **utils/**: Provides various utility functions, including data generation, Hessian matrix calculation, and neural network structure definitions.

- **train.py**: The main neural network training script used to train surrogate models in different examples.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/NN-HiSD.git
    cd NN-HiSD
    ```

2.  **Create and activate the Conda environment:**
    The `requirements.txt` file contains all the necessary dependencies.
    
    ```bash
    conda create -n nnhisd python=3.11
    conda activate nnhisd
    pip install torch --index-url=https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    ```
    
    **Note:** For GPU acceleration during neural network training, ensure your system supports CUDA 12.1. Alternatively, you can select an appropriate PyTorch version compatible with your GPU architecture by visiting https://pytorch.org/get-started/locally/.

## Reproducing the Experiments

**Important:** All commands below should be run from the root directory of the project.

## Example 1: Simple 2D/3D Model

This example demonstrates the core functionality on simple 2D and 3D potentials.

1. **Train the Surrogate Model:**
   ```bash
   python train.py examples/ex1_simple/simple2d.json
   ```

2. **Analyze Surrogate Quality:**
   Calculate and plot order-1 and order-2 derivative differences over training epochs.
   ```bash
   python examples/ex1_simple/epsilons_calculate.py
   ```

3. **Run HiSD and Compare Trajectories:**
   Visualize optimization paths on both exact and surrogate models.
   ```bash
   python examples/ex1_simple/traj.py
   ```

4. **Additional Analyses:**
   - Compare dimer method vs. AD for Hessian calculation: `python examples/ex1_simple/diff.py`
   - Test robustness with Gaussian noise in training data: `python examples/ex1_simple/addnoise.py`
   - Train for 3D potential: `python train.py examples/ex1_simple/parametric3d.json`
   - Run HiSD for 3D model (α=3, α=7):
     ```bash
     python examples/ex1_simple/exactSL.py
     python examples/ex1_simple/surrogateSL.py
     ```
   - Plot solution landscape for 3D model:
     ```bash
     python examples/ex1_simple/tree1.py
     python examples/ex1_simple/tree2.py
     ```
     
---

## Example 2: Muller-Brown Potential

This example analyzes neural network surrogate models for Muller-Brown potential systems.

1. **Train the Surrogate Model:**
   ```bash
   python train.py examples/ex2_mb/mb.json
   ```

2. **Compare Trajectories:**
   Visualize HiSD path between exact and surrogate models.
   ```bash
   python examples/ex2_mb/traj.py
   ```

3. **Gradient Impact Analysis:**
   Compare saddle point errors with and without gradient information.
   ```bash
   python examples/ex2_mb/mb_grads.py      # With gradients
   python examples/ex2_mb/mb_nograds.py    # Without gradients
   python examples/ex2_mb/ploterror.py     # Plot the picture
   ```

4. **Additional Analyses:**
   - Compare HiSD parameters: `python examples/ex2_mb/mbdiff.py`
   - Modified Muller-Brown potential:
     ```bash
     python train.py examples/ex2_mb/mmb.json
     python examples/ex2_mb/mmbdiff.py
     ```

---

## Example 3: Rosenbrock Function

This example demonstrates surrogate model performance with varying training data sizes on the Rosenbrock function.

1. **Train with Different Point Numbers:**
   ```bash
   python train.py examples/ex3_rosenbrock/rosenbrock.json
   ```

2. **Calculate Saddle Points:**
   Analyze performance with different numbers of training points.
   ```bash
   python examples/ex3_rosenbrock/rosennum.py
   ```

3. **Additional Analyses:**
   - Compare HiSD parameters: `python examples/ex3_rosenbrock/diff.py`
   - Gaussian Process Regression comparison (10k points): `python examples/ex3_rosenbrock/GPR.py`

---

## Example 4: Alanine Dipeptide

This example demonstrates the application of NN-HiSD to molecular dynamics simulations using NAMD for the alanine dipeptide system, a standard benchmark for biomolecular simulations.

### Prerequisites
To reproduce the data generation and comparative analyses, NAMD must be installed. Please download and install NAMD from the official website: https://www.ks.uiuc.edu/Research/namd/.

### 1. Generate Training Data
Molecular dynamics simulations can be performed using NAMD to generate training data. Alternatively, precomputed data is available in the `examples/ex4_nanma/vacuum.pmf` file for immediate use.

```bash
/path/to/NAMD/namd2 +p32 ./examples/ex4_nanma/MDparams/nnhisd/vacuum.conf > ./examples/ex4_nanma/MDparams/nnhisd/vacuum.log

copy ./examples/ex4_nanma/MDparams/nnhisd/output/vacuum.pmf ./examples/ex4_nanma
```

**Note:** Replace `/path/to/NAMD` with the actual installation path of NAMD on your system.

### 2. Train the Neural Network Surrogate Model

```bash
python examples/ex4_nanma/training.py
```

This script trains a neural network surrogate model based on the PMF data, optimizing to accurately approximate both the energy landscape and its derivatives.

### 3. Calculate Saddle Points

```bash
python examples/ex4_nanma/saddle.py
```

This command applies the HiSD algorithm to the trained surrogate model to identify and characterize saddle points in the energy landscape.

### 4. Visualization
Generate visual representations of the energy landscape and solution landscape:

```bash
python examples/ex4_nanma/energy.py     # Visualize energy landscape
python examples/ex4_nanma/tree.py       # Visualize solution landscape
```

### 5. Comparative Analyses
Perform comparative analyses with alternative approaches:

- **Original HiSD with Finite Difference Gradients:**
  ```bash
  python examples/ex4_nanma/original.py /path/to/NAMD/
  ```
  
- **Gaussian Process Regression Comparison:**
  ```bash
  python examples/ex4_nanma/gpr.py /path/to/NAMD/
  ```

**Note:** Replace `/path/to/NAMD` with the actual installation path of NAMD on your system.

---

## Example 5: Bacterial Ribosomal Assembly Intermediates

This example analyzes experimental bacterial system data using surrogate models.

**Data Files:** `examples/ex5_bacterial/num.txt`, `examples/ex5_bacterial/tsne_s28.npy`

1. **Visualize Experimental Data:**
   Plot piecewise-linear interpolation of experimental data.
   ```bash
   python examples/ex5_bacterial/energy1.py
   ```

2. **Train Surrogate Model:**
   ```bash
   python examples/ex5_bacterial/training.py
   ```

3. **Calculate Saddle Points:**
   ```bash
   python examples/ex5_bacterial/saddle.py
   ```

4. **Visualization:**
   Plot energy and solution landscapes of the surrogate model.
   
   ```bash
   python examples/ex5_bacterial/energy2.py     # Energy landscape
   python examples/ex5_bacterial/tree.py        # Solution landscape
   ```

## Citing this work

Yuankai Liu, Lei Zhang, Jin Zhao, "Neural Network-based High-index Saddle Dynamics Method for Searching Saddle Points and Solution Landscape".
arXiv preprint arXiv:2411.16200 (2024).

```
@article{Liu2024NNHiSD,
  title={Neural Network-based High-index Saddle Dynamics Method for Searching Saddle Points and Solution Landscape},
  author={Yuankai Liu, Lei Zhang, Jin Zhao},
  journal={arXiv preprint arXiv:2411.16200},
  year={2024}}
```

