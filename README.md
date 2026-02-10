# Improving Access to Essential Medicines via Decision-Aware Machine Learning 
[![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)](#required-software)
[![R](https://img.shields.io/badge/R-4.2.1-276DC3?logo=r&logoColor=white)](#required-software)
[![Gurobi](https://img.shields.io/badge/Gurobi-required-red)](#required-software)


This repository contains the **pre-deployment experiment** code and **baseline comparison** code used in **Supplementary Section S1.5.3**.

- Code is organized under: `./Experiment/`
- Required software: **Python 3.9** and **R 4.2.1**
- Notes:
  - Several scripts solve optimization problems and may require the **Gurobi** solver. Some scripts can be adapted to use **SciPy**.
  - Machine learning models may require **hyperparameter tuning** to obtain results similar to the paper.

---

## Repository Structure

Key files/folders under `Experiment/`:

- `requirements.txt` — Python dependencies
- `Preprocess.py` — preprocessing to create training data
- `BaselineComparison.R` — compares results across methods
- `DecisionAware/` — **Decision-aware** (our approach)
- `DecisionBlind/` — predict-then-optimize baseline
- `Excel/` — Excel tool baseline inputs/scripts
- `Distribution/` and `model_distribution.py` — distribution modeling baseline
- `Global Health (3 Month Rolling Avg)/` — 3-month rolling average baseline
- `StochOptForest/` — StochOptForest baseline adapted from Kallus & Mao (2023): Stochastic Optimization Forests, *Management Science*, 69(4), 1975–1994.


---

## Data

All required data are provided as a **GitHub Release asset** named:

- **`data.zip`**

### Download (GitHub UI)
1. Go to the repository page on GitHub.
2. Click **Releases** (right sidebar or top menu).
3. Open the latest release and download **`data.zip`** under **Assets**.
4. Unzip **into the `Experiment/` folder** so files end up at paths like:
   - `Experiment/df4ml.csv`
   - `Experiment/dfRaw_2023Q2.csv`
   - `Experiment/budget2023Q2Scipt_processed.csv`
   - `Experiment/S1_master_facility_update_11.csv`

---

## Requirements

- **Python 3.9**
- **R 4.2.1**
- **Gurobi** (for optimization scripts that use `gurobipy`)
  - Some scripts may support **SciPy** as an alternative solver (see script comments).

---

## Setup

### 1) Install Python dependencies
From the repository root:

```bash
pip install -r Experiment/requirements.txt
```

### 2) Install R dependencies
Open `Experiment/BaselineComparison.R` in RStudio and install any required packages listed at the top of the script:

```r
install.packages(c("data.table","stringr","ggplot2","viridis"))
```

---

## Reproducing Results (Supplementary S1.5.3)

### Step 0 — Preprocess (create training data)

```bash
cd Experiment
python Preprocess.py
```

---

## Step 1 — Generate results for each approach

Each method outputs a **facility × recommended allocation** table (CSV). We then conduct a **off-policy evaluation** to compare approaches by applying each method’s recommended allocations to historical demand/consumption data and computing performance over a **three-month horizon** (i.e., the three months within the quarter).

Below are the standard commands used for the setup:
- Dates (three-month window): `--date 2022-12-01 2023-01-01 2023-02-01`
- Budget setting: `--budgetType Real25`, which sets the **total supply (budget)** to the **25th percentile** of historical quarterly budgets for each product. **Note:** budget data in the shared dataset are **perturbed** to comply with the data-sharing agreement.


### (a) Decision-Aware (Our approach)
**Code:** `Experiment/DecisionAware/DAPriorQ2.py`

```bash
cd Experiment/DecisionAware
python DAPriorQ2.py --date 2022-12-01 2023-01-01 2023-02-01 --budgetType Real25
```

**Output:** `Experiment/DecisionAware/result/`

---

### (b) Decision-Blind (Predict-then-optimize)
**Code:** `Experiment/DecisionBlind/DBQ2.py`

```bash
cd Experiment/DecisionBlind
python DBQ2.py --date 2022-12-01 2023-01-01 2023-02-01 --budgetType Real25
```

**Output:** `Experiment/DecisionBlind/result/`

---

### (c) Excel tool
**Code:** `Experiment/BaselineComparison.R` (section `# Excel`)  
**Required data is under:** `Experiment/Excel/`

Run `BaselineComparison.R` in R/RStudio (see Step 2 below).

---

### (d) Distribution modeling
**Code:** `Experiment/model_distribution.py`

```bash
cd Experiment
python model_distribution.py --date 2022-12-01 2023-01-01 2023-02-01
```

**Output:** `Experiment/Distribution/results/`

---

### (e) Global Health (3 Month Rolling Avg)
**Code:** `Experiment/Global Health (3 Month Rolling Avg)/main_3mthAvg.py`

```bash
cd "Experiment/Global Health (3 Month Rolling Avg)"
python main_3mthAvg.py --date 2022-12-01 2023-01-01 2023-02-01
```

**Output:** `Experiment/Global Health (3 Month Rolling Avg)/results/`

---

### (f) StochOptForest
**Code & Steps:** `Experiment/StochOptForest/`

```bash
cd Experiment/StochOptForest
python getDual.py
python LoopNMSAPaper.py
```

Then:
1. Convert the output from `LoopNMSAPaper.py` to a CSV named:
   - `AllPdQ2_StochForestAllegro.csv`
2. Save it under:
   - `Experiment/StochOptForest/tmp/AllPdQ2_StochForestAllegro.csv`

Finally, run the allocation script:

```bash
python stochForest.py --date 2022-12-01 2023-01-01 2023-02-01 --budgetType Real25
```
> **Note (StochOptForest adaptation):** StochOptForest was originally proposed for settings where each training example corresponds to a standalone optimization problem. Our application has a different structure (a single product-level allocation problem coupled across many facilities and time periods), so we **adapted** the method to be compatible with our pipeline. As a result, observed performance gap could be reflecting *setting mismatch and adaptation constraints*. 


---

## Step 2 — Compare results across methods

After generating results for (a)–(f), run:

- `Experiment/BaselineComparison.R`

This script reads outputs from each method and produces the baseline comparison results reported in Supplementary **S1.5.3**.

---

## Troubleshooting / Notes

- **Gurobi:** If scripts use `gurobipy`, you need a working Gurobi installation + license. Some scripts may allow you to use SciPy as a fallback.

- Note that the data are perturbed to comply with data-sharing requirements. Reproducing similar results may also require fixing random seeds and performing hyperparameter tuning.

---

## Citation

If you use this code, please cite our paper:

```bibtex
@article{ChungetalAllocMedSL,
  title   = {Improving Access to Essential Medicines via Decision-Aware Machine Learning},
  author  = {},
  journal = {},
  year    = {YYYY},
  doi     = {DOI}
}
```






