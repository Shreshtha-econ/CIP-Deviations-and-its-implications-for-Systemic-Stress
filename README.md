# 📘 Master Thesis: Novel Foreign Exchange Risk Indicator through modeling Covered Interest Parity Deviations

**Author**: Shreshtha Tripathi  
**Date**: May 2025  
**Institution**: University of Amsterdam

This repository accompanies my Master Thesis, which investigates systemic financial stress through the lens of **Covered Interest Parity (CIP) deviations**. The project introduces a novel **CIP Stress Measure** and analyzes its explanatory power over the **ECB Composite Indicator of Systemic Stress (CISS)**.

---

## 🧾 Data Description

The analysis uses daily financial market data from central banks and financial databases for 5 currencies. For example,

- `y`: 3-Month U.S. Treasury Bill Rate (% per annum)
- `y*`: 3-Month EURIBOR (% per annum)
- `s`: USD/EUR Spot Exchange Rate
- `f`: 3-Month Forward USD/EUR Exchange Rate

Various indicators are used to capture CISS components:
- Bond Market Risk
- Equity Market Risk
- Foreign Exchange Market Risk
- Intermediary Market Risk
- Money Market Risk

---

## 📈 Methodology

### 1. CIP Deviation Calculation

CIP deviation \( x_t \) is defined as:

x_t = y_t - y^*_t - \log(F_t/S_t)


It quantifies arbitrage opportunities or distortions in the cross-currency basis market.

---

### 2. Lagged Features and PCA

- A lagged CIP deviation term \( x_{t-1} \) captures time dynamics.
- **Principal Component Analysis (PCA)** reduces the dimensionality of the CISS input space.

---

### 3. Kernel Quantile Regression (KQR)

- Models conditional quantiles of \( x_t \) based on principal components.
- Defines a **neutral band** using quantiles at τ = 0.05 and τ = 0.95.

---

### 4. CIP Stress Measure

CIP_Stress_t =
|x_t - Q̂_0.95(x_t)| if x_t > Q̂_0.95(x_t)
|x_t - Q̂_0.05(x_t)| if x_t < Q̂_0.05(x_t)
0 otherwise


- Transformed as: `Log CIP Stress = log(1 + CIP_Stress_t)`

---

### 5. Regression on ECB CISS

Standard portfolio theory is used to create a systemic risk indicator which gives weight to individual markets depending on how interconnected they are with other markets - thus capturing systemic risk. This is consistent with ECB's Composite Indicator of Systemic Stress (CISS). The indicator constructed leads the official CISS by nearly 3 months!


--

## 🔧 Installation & Usage

### Requirements

- Python 3.8+
- NumPy, pandas, scikit-learn, statsmodels
- seaborn, matplotlib

### To Run the Analysis

```bash
git clone https://github.com/your-username/your-thesis-repo.git
cd your-thesis-repo
pip install -r requirements.txt
python main.py
🧾 Citation

If you use this work, please cite:

Tripathi, Shreshtha. "Novel Foreign Exchange Risk Indicator through modeling Covered Interest Parity Deviations." Master Thesis, [University Name], June 2025.


For questions, reach out at: shreshtha453@gmail.com




---


