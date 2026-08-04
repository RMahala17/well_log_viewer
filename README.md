# PetroPy: Interactive Well Log Viewer & AI Formation Evaluation

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://well-log-viewer-f6uivdhzhz8izusmqdj5c7.streamlit.app/)
> **Live Web Application:** [well-log-viewer.streamlit.app](https://well-log-viewer-f6uivdhzhz8izusmqdj5c7.streamlit.app/)

An interactive web application built for automated parsing, normalization, multi-track visualization, and machine-learning-driven curve synthesis of standard `.las` well-log data. Designed to streamline subsurface workflows for reservoir characterization and petrophysical evaluation.

---

## Key Features

* **LAS Data Ingestion & Parsing:** Fast loading and structure validation of standard LAS (Log ASCII Standard) files using `lasio`.
* **Interactive Multi-Track Display:** Dynamic rendering of standard petrophysical tracks:
  * **Track 1:** Gamma Ray (GR) & Caliper (CALI) with lithology shading.
  * **Track 2:** Deep/Medium/Shallow Resistivity (LLD, LLS, MSFL) on logarithmic scales.
  * **Track 3:** Porosity Logs (Density `RHOB`, Neutron `NPHI`, Sonic `DT`) with crossover highlights.
* **Automated Formation Evaluation:**
  * Volume of Shale estimation using linear and non-linear (Larionov) equations.
  * Porosity calculations and Effective Porosity .
  * Water Saturation modeling via Archie's Equation.
* **AI/ML Curve Reconstruction:**
  * Built-in `Scikit-Learn` Random Forest Regressor pipeline to synthesize missing or corrupted log curves (e.g., predicting `DT` or `RHOB` from available tracks).
* **Automated Export & Archiving:**
  * Dynamic PDF petrophysical summary report generation (`FPDF`).
  * High-performance dataset storage via `Parquet` format for rapid retrieval.

---

## Benchmark Case Study: Ichthys Deep-1

The dashboard includes evaluation models benchmarked on public wireline log data from the **Ichthys Deep-1** gas-condensate discovery well (Browse Basin, Western Australia).

* **Target Formation:** Brewster Member (Deep Marine Sandstone Reservoir).
* **Key Analyses:** Net pay identification, shale-volume cutoffs, and fluid-type crossover verification.

---

## Tech Stack & Dependencies

| Category | Tools / Libraries |
| :--- | :--- |
| **Language** | Python 3.9+ |
| **Frontend/UI** | Streamlit |
| **Subsurface Tools** | Lasio, SciPy |
| **Data & ML** | Pandas, NumPy, Scikit-Learn |
| **Visualization** | Plotly, Matplotlib |
| **Reporting & Data** | FPDF, PyArrow (Parquet) |

---

## Local Installation & Setup

To run this repository locally on your machine:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/RMahala17/well-log-viewer.git](https://github.com/RMahala17/well-log-viewer.git)
   cd well-log-viewer
