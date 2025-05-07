# 📘 Master Thesis: Analyzing Systemic Stress via Covered Interest Parity Deviations

**Author**: Shreshtha Tripathi  
**Date**: May 2025  
**Institution**: [University of Amsterdam]

This repository accompanies my Master Thesis, which investigates systemic financial stress through the lens of **Covered Interest Parity (CIP) deviations**. The project introduces a novel **CIP Stress Measure** and analyzes its explanatory power over the **ECB Composite Indicator of Systemic Stress (CISS)**.

---

## 🧾 Data Description

The analysis uses daily financial market data from central banks and financial databases:

- `y`: 3-Month U.S. Treasury Bill Rate (% per annum)
- `y*`: 3-Month EURIBOR (% per annum)
- `s`: USD/EUR Spot Exchange Rate
- `f`: 3-Month Forward USD/EUR Exchange Rate

CISS components (excluding Money Market input):
- Bond Market Risk
- Equity Market Risk
- Foreign Exchange Market Risk
- Intermediary Market Risk

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

A linear regression is used to evaluate the link between the stress measure and systemic risk:

CISS_t = β₀ + β₁·Band_t + β₂·Bond_t + β₃·Equity_t + β₄·FX_t + β₅·Intermediaries_t + ε_t


---

## 📊 Key Results

- **Intermediaries Input** and **Band Width** are significant predictors of systemic stress.
- High **R² = 0.780**, indicating strong explanatory power.
- Highlights the importance of market frictions in assessing financial stability.

---

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

Tripathi, Shreshtha. "Analyzing Systemic Stress via Covered Interest Parity Deviations." Master Thesis, [University Name], May 2025.
📩 Contact

For questions, reach out at: [your-email@example.com]

📄 License

MIT License. See LICENSE file for details.


---

Let me know if you’d like help setting up the code structure (`main.py`, data folders, etc.) or generating plots for the README.
