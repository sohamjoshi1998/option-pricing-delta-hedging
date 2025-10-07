# 🧮 Option Pricing and Delta Hedging using Monte Carlo Simulation

**Independent Project**  
**Tools:** Python (`NumPy`, `Pandas`, `Matplotlib`), C++  
**Author:** Soham Joshi  

---

## 📘 Overview

This project implements a **Monte Carlo simulator** to price **European and Asian options** under the **Black–Scholes–Merton (BSM)** model and analyzes **delta-hedging strategies** to measure hedging error under discrete rebalancing.  
It also extends the implementation to **C++** for computational efficiency and compares results with the Python version.

---

## 🎯 Objectives

- ✅ Price **European Call/Put options** using Monte Carlo simulation.  
- ✅ Extend the model to **Asian options** (arithmetic average).  
- ✅ Compare simulated results with the **analytical BSM model**.  
- ✅ Implement **delta-hedging** and analyze **hedging error** over different rebalancing frequencies.  
- ✅ Port the code to **C++** to benchmark performance.

---

## 🧠 Theoretical Background

### 1. Black–Scholes–Merton (BSM) Model

Under the risk-neutral measure, the stock price \( S_t \) follows a **Geometric Brownian Motion**:

\[
dS_t = r S_t dt + \sigma S_t dW_t
\]

The analytical price of a **European Call** is:

\[
C = S_0 N(d_1) - K e^{-rT} N(d_2)
\]

where  

\[
d_1 = \frac{\ln(S_0/K) + (r + 0.5\sigma^2)T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}
\]

---

### 2. Monte Carlo Pricing

We simulate \( N \) possible future paths for \( S_T \):

\[
S_T = S_0 e^{(r - 0.5\sigma^2)T + \sigma \sqrt{T}Z}
\]

Then, the Monte Carlo price is:

\[
C_{MC} = e^{-rT} \cdot E[\max(S_T - K, 0)]
\]

**Variance reduction:** Implemented using *Antithetic Variates*.  
Typical simulation count: **30,000+** for stable results.

---

### 3. Asian Option Extension

For **Asian options**, the payoff depends on the average of simulated prices over time:

\[
C_{Asian} = e^{-rT} E[\max(\bar{S} - K, 0)], \quad \bar{S} = \frac{1}{n} \sum_{i=1}^n S_{t_i}
\]

This makes them path-dependent and more computationally intensive.

---

### 4. Delta Hedging

A **delta-hedged portfolio** involves:

- Holding **one option** and  
- **Shorting Δ shares** of the underlying asset.

As the underlying price changes, we rebalance Δ periodically (daily, weekly, monthly, yearly).  
Due to discrete rebalancing, the hedge is imperfect — this leads to **hedging error**.

We measure hedging error as the difference between the **final portfolio value** and **option payoff** at maturity.

---

## ⚙️ Implementation Steps

### Step 1: Parameter Setup
Define inputs:

S0 = 100     # Initial stock price
K = 100      # Strike price
r = 0.05     # Risk-free rate
sigma = 0.2  # Volatility
T = 1        # Time to maturity (years)

### Step 2: Monte Carlo Simulation (Python)

Simulate stock price paths using Geometric Brownian Motion.
Compute discounted payoffs for European and Asian options.
Use antithetic variates to reduce variance.
Compare Monte Carlo results vs. BSM analytical price.

### Step 3: Delta Hedging

Compute delta using the BSM delta formula at each step.
Rebalance portfolio across different frequencies.
Measure Mean Error, Standard Deviation, and MAE of hedging performance.
Frequency	Steps	Mean Error	Std Dev	MAE
Yearly	1	-0.13	5.91	4.75
Monthly	12	-0.04	1.94	1.47
Weekly	52	-0.02	0.96	0.72
Daily	252	~0.00	0.44	0.33

✅ Observation: Increasing rebalancing frequency reduces hedging error, approaching a perfect hedge.

### Step 4: C++ Implementation

Rewrote the Monte Carlo pricer using <random> and <cmath> for high-speed execution.
Used std::vector for path generation and payoff computation.
Observed significant speedup (3–5×) for large simulations.
To compile and run:
g++ option_pricing.cpp -o option_pricing
./option_pricing

📊 Results Summary

Metric	Analytical	Monte Carlo	Error
European Call	8.433	8.444 ± 0.078	0.12%
Asian Call (Arithmetic)	—	4.870 ± 0.086	—
Pricing error: <5% (validated model correctness)
Hedging error decreases with rebalance frequency
C++ version runs faster, maintaining same accuracy

📈 Visualizations

The project generates several visual plots:
Simulated Stock Price Paths
Distribution of Terminal Prices
Hedging Error Convergence

🧩 Project Structure
OptionPricingMonteCarlo/
│
├── README.md                     # Project documentation
├── option_pricing.py             # Python Monte Carlo simulator
├── delta_hedging.py              # Delta hedging analysis
├── option_pricing.cpp            # C++ version for benchmarking
├── results/
│   ├── price_paths.png
│   ├── terminal_distribution.png
│   ├── hedging_error.png
│   └── report.pdf
└── requirements.txt              # Python dependencies

🧰 Dependencies
Install required packages:
pip install numpy pandas matplotlib scipy

💡 Key Insights
Monte Carlo simulation provides flexibility and accuracy in pricing non-analytical options.
Hedging performance improves as rebalancing frequency increases.
Variance reduction techniques like Antithetic Sampling significantly stabilize estimates.
C++ implementation boosts computation speed for large-scale simulations.

🧑‍💻 Author
Soham Joshi
Master of Financial Mathematics (MFM) Candidate – NCSU
Independent Quantitative Finance Project
📧 Email: ssjosh22@ncsu.edu /
soham.joshi.work@gmail.com
🔗 LinkedIn - https://www.linkedin.com/in/sohamjoshi1998/

