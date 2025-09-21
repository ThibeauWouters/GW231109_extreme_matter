# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a gravitational wave (GW) parameter estimation project focused on S250818k/GW231109 extreme matter investigations. The codebase performs Bayesian inference on gravitational wave events using various waveform models (IMRPhenomXP_NRTidalv3, IMRPhenomXAS_NRTidalv3) to constrain neutron star equation of state parameters.

## Key Architecture Components

### Core Analysis Pipeline
- **`utils.py`**: Central utilities for posterior analysis, including functions to fetch sampling times, log Bayes factors, waveform approximants, and posterior samples from HDF5/JSON files
- **`make_overview_table.py`**: Automated analysis script that processes multiple inference runs and generates summary tables with run metadata (sampling times, Bayes factors, priors used)

### Inference and Modeling (`jester/` directory)
- **`jester/inference.py`**: Full-scale Bayesian inference using jim (flowMC wrapper) with JAX backend for GPU acceleration
- **`jester/utils.py`** and **`jester/utils_plotting.py`**: Specialized utilities for the inference pipeline and plotting
- **`jester/postprocessing.py`**: Post-processing of inference results
- **`jester/train_NF.py`**: Training normalizing flow models

### Neural Priors (`neural_priors/` directory)
- **`neural_priors/train_NF_prior.py`**: Training normalizing flows to approximate mass-Lambda distributions using glasflow and PyTorch, with bilby integration for GW parameter estimation
- **`neural_priors/evaluate_flows.py`**: Evaluation of trained neural flow models

### Visualization (`figures/` directory)
- **`figures/plot_mass_Lambdas.py`**: 2D contour plots of mass vs Lambda parameters
- **`figures/plot_corners.py`**: Corner plots for posterior distributions
- **`figures/make_presentation_figs.py`**: Generation of publication-quality figures
- **`figures/reweigh_lambda_tilde.py`**: Reweighting of tidal deformability parameters
- **`figures/utils.py`**: Plotting utilities

### Data Processing
- **`PSD/`**: Power spectral density analysis scripts
- **`posteriors/`**: Posterior sample processing and marginalization
- **`em_lightcurves/`**: Electromagnetic counterpart analysis

## Key Dependencies

- **JAX**: High-performance computing backend with GPU support
- **jim/jimgw**: Gravitational wave inference framework with flowMC
- **bilby**: Bayesian inference library for gravitational wave astronomy
- **glasflow**: Normalizing flows for density estimation
- **PyTorch**: Deep learning framework for neural network training
- **h5py**: HDF5 file handling for posterior samples
- **corner**: Plotting library for parameter estimation visualization

## Common Development Tasks

### Running Single Analysis
Execute the main inference script:
```bash
python jester/inference.py --make-cornerplot True
```

### Processing Multiple Runs
Generate overview table for all runs:
```bash
python make_overview_table.py
```

### Training Neural Priors
Train normalizing flow priors:
```bash
python neural_priors/train_NF_prior.py
```

### Plotting Results
Generate mass-Lambda plots:
```bash
python figures/plot_mass_Lambdas.py
```

## Data Structure

- **`overview/all_runs_information.json`**: Metadata for all inference runs including priors, sampling times, and Bayes factors
- **`posteriors/data/`**: Processed posterior samples
- **`figures/EOS_data/`**: Equation of state data for neutron stars
- Inference outputs are stored in `outdir/final_result/` with HDF5 posterior files

## Project Context

This repository analyzes the GW231109 gravitational wave event, investigating extreme matter properties through tidal deformability measurements. The analysis includes:
- Comparison of different waveform approximants (aligned vs precessing spin models)
- Various prior configurations for chirp mass, mass ratio, spin, and Lambda parameters
- Neural normalizing flow priors trained on equation of state data
- Systematic studies across 70+ different analysis configurations

## Detailed Plotting Scripts Documentation

The repository contains sophisticated plotting infrastructure for visualizing gravitational wave parameter estimation results and equation of state constraints.

### Core Plotting Scripts (`figures/` directory)

#### `figures/plot_mass_Lambdas.py`
**Purpose**: Creates mass-Lambda contour plots overlaying GW parameter estimation results on equation of state curves.

**Key Functions**:
- `make_plot_chirp_tilde()`: Main plotting function for chirp mass vs Lambda tilde plots
- `make_plot_components()`: Alternative view showing component masses vs individual Lambdas (deprecated)
- `load_eos_curves()`: Loads EOS data from NPZ files containing mass, radius, Lambda arrays
- `get_mchirp_lambda_tilde_EOS()`: Converts EOS mass-Lambda curves to chirp mass-Lambda tilde space
- `fetch_prior_samples()`: Generates prior samples for comparison

**Features**:
- Overlays posterior contours from GW231109 and GW170817 events
- Color-coded EOS curves based on posterior probability
- Support for both tabular EOS data and Jester inference results
- Automatic legend generation with event identification
- Configurable mass and Lambda ranges via command line arguments

**Usage**: `python figures/plot_mass_Lambdas.py --nb-samples 10000 --mass-max 2.0`

#### `figures/plot_corners.py`
**Purpose**: Generates corner plots (parameter correlation plots) for inference runs.

**Key Functions**:
- `make_cornerplot()`: Creates corner plots from HDF5 posterior files
- Automatically extracts prior information from result files
- Handles different analysis types (fixed distance/sky location, quasi-universal relations)

**Features**:
- Loads sampling metadata (log Bayes factor, sampling time)
- Automatically filters irrelevant parameters based on run type
- Supports additional derived parameters (chi_eff, lambda_tilde)
- Person-specific directory organization and naming conventions
- Error handling for problematic parameter combinations

**Usage**: `python figures/plot_corners.py --overwrite`

#### `figures/make_presentation_figs.py`
**Purpose**: Specialized script for creating publication-quality comparison plots.

**Key Functions**:
- `load_hdf5()` / `load_json()`: Load posterior samples from different file formats
- Creates overlaid corner plots comparing multiple analysis runs
- Generates combined parameter distribution plots

**Features**:
- Hardcoded file paths for specific high-priority comparisons
- Support for both HDF5 and JSON posterior formats
- Automated legend generation for multi-run comparisons
- High DPI output for presentation use

#### `figures/plot_m1m2_comparison.py`
**Purpose**: Specialized 2D mass comparison plots across multiple inference runs.

**Key Functions**:
- `make_m1m2_comparison_plot()`: Creates component mass correlation plots
- Automatically determines plot limits from data quantiles
- Color-codes test runs vs production runs

**Features**:
- Batch processing of multiple analysis configurations
- Predefined comparison sets for systematic studies
- Dynamic plot range calculation based on data
- Legend management for multiple overlaid datasets

#### `figures/reweigh_lambda_tilde.py`
**Purpose**: Performs Savage-Dickey ratio calculations for Bayesian model comparison.

**Key Functions**:
- `reflected_kde()`: KDE with boundary correction for Lambda parameters
- Computes posterior/prior ratios for model evidence estimation
- Reweights posterior samples to flat Lambda priors

**Features**:
- Handles different prior types (uniform vs quasi-universal)
- Automatic Savage-Dickey ratio computation
- Generates reweighted posterior distributions
- Supports boundary-corrected density estimation

### Inference Plotting (`jester/utils_plotting.py`)

**Purpose**: Specialized plotting utilities for equation of state inference results.

**Key Functions**:
- `plot_corner()`: Corner plots for EOS parameter samples
- `plot_eos()`: Multi-panel EOS visualization (micro and macro physics)

**Features**:
- Microscopic EOS plots: pressure, energy density, sound speed vs density
- Macroscopic plots: mass-radius and mass-Lambda relationships
- Configurable sample overlay with transparency
- Physical unit conversions (geometric to physical units)

### Analysis Support Scripts

#### `neural_priors/train_NF_prior.py` and `neural_priors/evaluate_flows.py`
- Include matplotlib for training diagnostics and flow evaluation plots
- Generate loss curves, corner plots of trained distributions
- Model comparison visualizations

#### `PSD/make_psd.py` and `PSD/check_anna_psd.py`
- Power spectral density analysis and validation plots
- Detector noise characterization visualizations

### Plotting Configuration

**Common matplotlib settings across scripts**:
- Serif fonts with 16pt labels for publication quality
- Disabled text.usetex for compatibility (some scripts enable it)
- Consistent color schemes: GW231109 (red), GW170817 (orange), priors (gray)
- Corner plot defaults: 40 bins, smoothing, filled contours, no data points

**File organization**:
- Outputs organized by person (`anna`, `tim`, `thibeau`) and run type
- Separate directories for different plot types (`mass_Lambdas_figs/`, `cornerplots/`)
- PDF format for vector graphics, PNG for presentations

### Running Plotting Scripts

Most scripts support command-line arguments for batch processing:
```bash
# Generate mass-Lambda plots for all EOS models
python figures/plot_mass_Lambdas.py --overwrite --nb-samples 5000

# Create corner plots for all inference runs
python figures/plot_corners.py --overwrite

# Compare component masses across neural prior runs
python figures/plot_m1m2_comparison.py
```