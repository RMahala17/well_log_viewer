# AI Petrophysics Dashboard

An advanced **Streamlit-based petrophysics and well log analysis application** designed for visualization, processing, interpretation, and machine learning-assisted analysis of subsurface well log data.

The application allows users to upload **LAS (Log ASCII Standard) files**, analyze different petrophysical curves, perform log processing, generate crossplots, evaluate formations, train machine learning models, and interact with an AI-powered petrophysics assistant.

---

# Project Overview

Well logging data contains valuable information about subsurface formations, reservoir properties, and hydrocarbon potential. However, interpreting large LAS datasets requires multiple software tools.

**AI Petrophysics Dashboard** provides an integrated environment where geoscientists and students can:

- Load and visualize LAS well log files
- Analyze raw and processed log curves
- Perform petrophysical interpretation
- Compare wells using a local repository
- Apply machine learning models
- Generate analysis reports
- Use AI assistance for interpretation

---

# Key Features

## 1. LAS File Processing

- Upload custom LAS files
- Supports real-world LAS encoding variations
- Automatically extracts:
  - Well information
  - Curve information
  - Parameter information
- Handles depth-based datasets
- Provides interactive data tables

---

# 2. Interactive Well Log Visualization

### Recorded Logs Viewer

Features:

- Display multiple log curves simultaneously
- Adjustable curve settings
- Custom depth intervals
- Linear and logarithmic scales
- Custom axis ranges
- Interactive Plotly-based visualization


Supported curves include:

- Gamma Ray (GR)
- Resistivity (RT)
- Density (RHOB)
- Neutron Porosity (NPHI)
- Sonic (DT/DTS)
- Caliper (CALI)
- Spontaneous Potential (SP)
- Other standard petrophysical curves


---

# 3. Smoothed Log Analysis

The application provides curve smoothing functionality:

- Moving average filtering
- Raw vs smoothed curve comparison
- Adjustable smoothing window size
- Custom visualization settings

Useful for reducing noise and identifying formation trends.

---

# 4. Raw Data Engineering Tools

The Raw Data module provides spreadsheet-style editing capabilities:

Features:

- View complete log dataset
- Add custom log curves
- Delete unwanted curves
- Fill missing values
- Edit values interactively
- Maintain synchronized datasets

---

# 5. Multi-Track Well Log Viewer

A professional-style multi-track visualization system:

Features:

- Multiple track layouts
- Multiple curve plotting
- Depth synchronized display
- Custom track configuration

Designed similar to industry-standard well log interpretation software.

---

# 6. Petrophysical Crossplots

Interactive crossplot analysis for studying relationships between different log measurements.

Examples:

- Density-Neutron crossplots
- Sonic relationships
- Porosity analysis
- Lithology identification

---

# 7. Formation Evaluation Module

Provides tools for petrophysical interpretation:

Includes analysis of:

- Lithology indicators
- Porosity-related logs
- Reservoir properties
- Formation characteristics

---

# 8. Machine Learning Module

The application integrates machine learning workflows for curve prediction and analysis.

Implemented models:

- Random Forest Regression
- Gradient Boosting Regression
- Extra Trees Regression


Capabilities:

- Select target curves
- Select feature curves
- Train prediction models
- Evaluate model performance
- Predict missing curves


Evaluation metrics:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)

---

# 9. Well Repository System

The application includes a local well data repository.

Features:

- Store multiple LAS files
- Build a master subsurface database
- Normalize different curve naming conventions
- Compare similar wells

Supported curve families include:

- Gamma Ray
- Resistivity
- Density
- Neutron Porosity
- Sonic
- Caliper
- Porosity
- Water Saturation
- Permeability


---

# 10. Sister Well Matching

The system can identify statistically similar wells from the repository.

Matching is based on:

- Curve availability
- Statistical similarity
- Mean and variation comparison
- Shared petrophysical characteristics


---

# 11. AI Petrophysics Assistant

The application includes an AI assistant with two modes:


## Local AI Engine

Uses locally running AI models through Ollama.

Advantages:

- Runs locally
- Keeps data private
- No cloud dependency


Supported models:

- Llama 3.1
- Moondream


## Cloud AI Engine

Uses API-based AI assistance for deployed applications.

Capabilities:

- Petrophysical questions
- Data interpretation assistance
- Image-based analysis support

---

# 12. Automated Report Generation

The application supports report generation including:

- Well information
- Log analysis
- Visualization outputs
- Interpretation results

---

# Technology Stack

## Programming Language

- Python


## Framework

- Streamlit


## Data Processing

- Pandas
- NumPy
- LASIO


## Visualization

- Plotly


## Machine Learning

- Scikit-learn


## Additional Tools

- Parquet Database Storage
- Ollama AI Integration
- Anthropic API Integration

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/RMahala17/well_log_viewer.git

cd well_log_viewer
