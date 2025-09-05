# GTMKit

> Mapping high-dimensional biology & chemistry into intuitive, navigable spaces with **Generative Topographic Mapping (GTM)**.

<p align="left">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-brightgreen"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11+-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.7.1%2B-red">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-CUDA_Ready-9cf">
</p>

---

## Table of contents

<details>
<summary><strong>Expand</strong></summary>

* [Overview](#overview)
* [Key Features](#key-features)
* [Installation](#installation)

  * [Requirements](#requirements)
  * [Using PDM (Recommended)](#using-pdm-recommended)
  * [Using pip](#using-pip)
* [Quick Start](#quick-start)

  * [Basic GTM Training](#basic-gtm-training)
  * [Creating Density Landscapes](#creating-density-landscapes)
  * [Classification Landscapes](#classification-landscapes)
  * [Regression Landscapes](#regression-landscapes)
* [Advanced Features](#advanced-features)

  * [Responsibility Pattern (RP) Fingerprints](#responsibility-pattern-rp-fingerprints)
  * [Molecular Coordinate Calculation](#molecular-coordinate-calculation)
  * [Using Altair for Static Visualizations](#using-altair-for-static-visualizations)
* [Model Configuration](#model-configuration)

  * [GTM Parameters](#gtm-parameters)
  * [PCA Initialization Options (GTM class)](#pca-initialization-options-gtm-class)
* [Architecture](#architecture)

  * [Core Components](#core-components)
  * [Key Classes](#key-classes)
* [Performance Tips](#performance-tips)
* [Testing](#testing)
* [Development](#development)

  * [Contributing](#contributing)
* [Citation](#citation)
* [Applications of GTM](#applications-of-gtm)

  * [Biological datasets](#biological-datasets)
  * [Chemical space & big chemical data](#chemical-space--big-chemical-data)
* [Contributors](#contributors)
* [Acknowledgements](#acknowledgements)
* [License](#license)
* [Contact](#contact)

</details>

---

## Overview

GTMKit is a comprehensive Python library for exploring chemical space and high-dimensional data using **Generative Topographic Mapping (GTM)**. GTM is a probabilistic dimensionality reduction technique that creates non-linear mappings from high-dimensional data spaces to interpretable low-dimensional latent spaces using a generative model with radial basis functions. This is a **PyTorch-based** implementation of the GTM algorithm that runs on **GPU**, and includes functions for building landscapes and GTM-specific metrics.

> \[!TIP]
> Pair GTM maps with interactive notebooks or dashboards to let users zoom from global chemical space down to neighborhood-level structure–activity patterns.

---

## Key Features

* **GPU-Accelerated**: PyTorch-based implementation with CUDA support for fast computation
* **Multiple GTM Variants**:

  * `VanillaGTM`: Basic GTM implementation with random initialization
  * `GTM`: Enhanced version with PCA-based initialization for better convergence
* **Comprehensive Visualization**:

  * Interactive landscapes using Plotly (smooth heatmaps)
  * Static visualizations using Altair (discrete grid-based plots)
  * Support for density, classification, and regression landscapes
* **Advanced Analytics**:

  * Responsibility Patterns (RP) for chemical space coverage analysis
  * Classification and regression landscape analysis

---

## Installation

### Requirements

* Python ≥ 3.11
* PyTorch ≥ 2.7.1
* NumPy ≥ 2.3.2
* Pandas ≥ 2.3.2
* Scikit-learn ≥ 1.7.1
* Altair ≥ 5.5.0
* Plotly ≥ 6.3.0
* tqdm ≥ 4.67.1

### Using PDM (Recommended)

```bash
git clone <repository-url>
cd GTMKit
pdm install
```

### Using pip

```bash
pip install numpy>=2.3.2 torch>=2.7.1 pandas>=2.3.2 altair>=5.5.0 plotly>=6.3.0 scikit-learn>=1.7.1 tqdm>=4.67.1
```

> \[!NOTE]
> For GPU acceleration, install a CUDA-enabled build of PyTorch appropriate for your system.

---

## Quick Start

### Basic GTM Training

```python
import torch
import numpy as np
from gtmkit.gtm import GTM

# Generate sample data
data = torch.randn(1000, 50, dtype=torch.float64)  # 1000 samples, 50 features

# Create GTM model
gtm = GTM(
    num_nodes=100,           # 10x10 grid in latent space
    num_basis_functions=25,  # 5x5 RBF centers
    basis_width=0.3,         # RBF width parameter
    reg_coeff=0.01,          # Regularization coefficient
    device="cuda"            # Use GPU if available
)

# Fit model and transform data
latent_coords = gtm.fit_transform(data)
print(f"Latent coordinates shape: {latent_coords.shape}")  # (2, 1000)

# Get responsibilities for landscape analysis
responsibilities, log_likelihoods = gtm.project(data)
print(f"Responsibilities shape: {responsibilities.shape}")  # (100, 1000)
```

### Creating Density Landscapes

```python
import numpy as np
from gtmkit.utils.density import get_density_matrix, density_to_table
from gtmkit.plots.plotly_landscapes import plotly_smooth_density_landscape

# Calculate density matrix
responsibilities_np = responsibilities.T.cpu().numpy()  # Convert to numpy
density = get_density_matrix(responsibilities_np)

# Create density table for visualization
density_table = density_to_table(density, node_threshold=0.1)

# Generate interactive Plotly landscape
fig = plotly_smooth_density_landscape(
    density_table,
    title="GTM Density Landscape",
    node_threshold=0.1
)
fig.show()
```

### Classification Landscapes

```python
from gtmkit.utils.classification import get_class_density_matrix, class_density_to_table
from gtmkit.plots.plotly_landscapes import plotly_discrete_class_landscape

# Sample binary classification labels
class_labels = np.random.choice([0, 1], size=1000)
class_names = ["Inactive", "Active"]

# Calculate class density matrices
density, class_density, class_prob = get_class_density_matrix(
    responsibilities_np,
    class_labels,
    class_name=class_names,
    normalize=True
)

# Create classification table
class_table = class_density_to_table(
    density, class_density, class_prob,
    node_threshold=0.0,  # Set to 0 for Plotly visualization
    class_name=class_names,
    normalized=True
)

# Generate classification landscape
fig = plotly_discrete_class_landscape(
    class_table,
    title="GTM Classification Landscape",
    first_class_label="Inactive",
    second_class_label="Active",
    min_density=0.1
)
fig.show()
```

### Regression Landscapes

```python
from gtmkit.utils.regression import get_reg_density_matrix, reg_density_to_table
from gtmkit.plots.plotly_landscapes import plotly_smooth_regression_landscape

# Sample regression values
regression_values = np.random.normal(5.0, 2.0, size=1000)

# Calculate regression density matrix
density, reg_density = get_reg_density_matrix(responsibilities_np, regression_values)

# Create regression table
reg_table = reg_density_to_table(
    density, reg_density,
    node_threshold=0.0  # Set to 0 for Plotly visualization
)

# Generate regression landscape
fig = plotly_smooth_regression_landscape(
    reg_table,
    title="GTM Regression Landscape",
    regression_label="Property Value",
    min_density=0.1
)
fig.show()
```

---

## Advanced Features

### Responsibility Pattern (RP) Fingerprints

```python
from gtmkit.metrics import resp_to_pattern, compute_rp_coverage

# Convert responsibilities to RP fingerprints
rp_fingerprints = np.array([
    resp_to_pattern(resp, n_bins=10, threshold=0.01)
    for resp in responsibilities_np
])

# Calculate coverage between datasets
reference_fps = rp_fingerprints[:500]  # First 500 as reference
test_fps = rp_fingerprints[500:]       # Last 500 as test

coverage = compute_rp_coverage(reference_fps, test_fps, use_weight=True)
print(f"Weighted coverage: {coverage:.3f}")
```

### Molecular Coordinate Calculation

```python
from gtmkit.utils.molecules import calculate_mols_coords

# Calculate molecular coordinates for plotting
mol_coords = calculate_mols_coords(
    responsibilities_np,
    correction=True,  # Adjust for visualization
    return_node=True  # Include most responsible node
)
print(mol_coords.head())
```

### Using Altair for Static Visualizations

```python
from gtmkit.plots.altair_landscapes import (
    altair_discrete_density_landscape,
    altair_discrete_class_landscape,
    altair_points_chart
)

# Create discrete density landscape
density_chart = altair_discrete_density_landscape(
    density_table,
    title="GTM Density Map"
)

# Overlay molecular points
points_chart = altair_points_chart(
    mol_coords,
    num_nodes=100,
    points_size=50,
    points_color="red"
)

# Combine charts
combined = density_chart + points_chart
combined.show()
```

---

## Model Configuration

### GTM Parameters

* **num\_nodes**: Number of latent space grid nodes (must be perfect square for 2D)
* **num\_basis\_functions**: Number of RBF centers (must be perfect square for 2D)
* **basis\_width**: RBF width parameter (controls smoothness)
* **reg\_coeff**: Regularization coefficient (prevents overfitting)
* **standardize**: Whether to standardize input data (recommended: True)
* **max\_iter**: Maximum EM algorithm iterations
* **tolerance**: Convergence tolerance
* **device**: Computation device ("cpu" or "cuda")

### PCA Initialization Options (GTM class)

* **pca\_engine**: PCA implementation ("sklearn" or "torch")
* **pca\_scale**: Scale eigenvectors by sqrt of eigenvalues
* **pca\_lowrank**: Use low-rank PCA approximation for large datasets

---

## Architecture

### Core Components

* **`gtm.py`**: Main GTM implementations (`BaseGTM`, `VanillaGTM`, `GTM`)
* **`metrics.py`**: RP fingerprints and coverage metrics
* **`utils/`**: Specialized analysis modules

  * `classification.py`: Binary/multi-class analysis
  * `regression.py`: Continuous property analysis
  * `density.py`: Density calculations and grid mapping
  * `molecules.py`: Molecular coordinate calculations
* **`plots/`**: Visualization modules

  * `plotly_landscapes.py`: Interactive smooth heatmaps
  * `altair_landscapes.py`: Static discrete visualizations

### Key Classes

* **`DataStandardizer`**: Robust data preprocessing with NaN handling
* **`BaseGTM`**: Abstract base class defining GTM interface
* **`VanillaGTM`**: Basic GTM with random initialization
* **`GTM`**: Enhanced GTM with PCA-based initialization

---

## Performance Tips

1. **Use GPU**: Set `device="cuda"` for significant speedup on large datasets
2. **Choose appropriate grid size**: Balance between resolution and computational cost
3. **PCA initialization**: Use `GTM` class instead of `VanillaGTM` for better convergence
4. **Data standardization**: Always enable for numerical stability
5. **Batch processing**: Process large datasets in chunks if memory is limited

---

## Testing

Run the comprehensive test suite:

```bash
pdm run pytest tests/ -v
```

Run with coverage:

```bash
pdm run pytest tests/ --cov=src/gtmkit --cov-report=html
```

---

## Development

### Contributors

* Tagir Akhemtshin
* Louis Plyer
* Alexey Orlov
* Alexandre Varnek


### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run quality checks: `pdm run pre-commit run --all-files`
5. Submit a pull request

---

## Citation

If you use this code please cite [future\_url](future_url)

---

## Applications of GTM

GTM has been applied across biological data and extensively studied for analyzing large chemical datasets and exploring chemical space, including virtual screening, library comparison/design, de novo compound design, and multi-scale visualization.

### Biological datasets

| Domain       | Representative study                                                                            |
| :----------- | :---------------------------------------------------------------------------------------------- |
| **Genomes**  | [Molecular Informatics (2023)](https://onlinelibrary.wiley.com/doi/full/10.1002/minf.202300263) |
| **Proteins** | [Bioinformatics (2022)](https://academic.oup.com/bioinformatics/article/38/8/2307/6528316)      |
| **Peptides** | [bioRxiv (2024)](https://www.biorxiv.org/content/10.1101/2024.11.17.622654v1.abstract)          |

### Chemical space & big chemical data

**Virtual screening**

* [Molecular Informatics (2018)](https://onlinelibrary.wiley.com/doi/abs/10.1002/minf.201800166)
* [European Journal of Medicinal Chemistry (2019)](https://linkinghub.elsevier.com/retrieve/pii/S0223-5234%2819%2930016-9)

**Library comparison & design**

* [Molecular Informatics (2011)](https://onlinelibrary.wiley.com/doi/abs/10.1002/minf.201100163)
* [Journal of Chemical Information and Modeling (2015)](https://pubs.acs.org/doi/10.1021/ci500575y)
* [PubMed (2019)](https://pubmed.ncbi.nlm.nih.gov/31407224/)
* [Molecular Informatics (2021)](https://onlinelibrary.wiley.com/doi/full/10.1002/minf.202100289)
* [JCIM (2023)](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.3c00520)

**De novo design of chemical compounds**

* [JCIM (2019)](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.8b00751)

**Multi-scale visualization of large chemical spaces**

* [JCIM (2022)](https://pubs.acs.org/doi/10.1021/acs.jcim.2c00509)

---


## Acknowledgements

The authors thank Dr. Arkadii Lin and Dr. Yuliana Zabolotna for their contributions to the development of the initial versions of the functions for density and classification landscape building, as well as GTM-derived metric calculation.

---

## License

This project is licensed under the MIT License — see the **LICENSE** file for details.

---

## Contact

Contact: [varnek@unistra.fr](mailto:varnek@unistra.fr)
