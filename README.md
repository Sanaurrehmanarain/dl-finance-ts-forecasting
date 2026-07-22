<div align="center">
  <a href="Report.pdf">
    <img src="dl1.png" alt="banner" width="100%">
  </a>
  <p><em>Click the banner to view the full analysis report</em></p>
</div>

# AAPL Time Series Forecasting: Stationarity, Fractional Differencing, MLP vs CNN (GAF)

A comprehensive time series analysis and forecasting project comparing traditional lag-based MLP models with CNN models trained on Gramian Angular Field (GAF) image encodings for **Apple Inc. (AAPL)** daily adjusted close prices.

---

## 📊 Project Overview

This project explores three distinct time series representations:

1. **Price Levels** (non-stationary)
2. **Log Returns** (stationary candidate)
3. **Fractionally Differenced Levels** (stationarity–memory tradeoff)

Two model architectures are systematically compared:

- **MLP (Multi-Layer Perceptron)** — Using lag features for direct time series prediction
- **CNN (Convolutional Neural Network)** — Trained on GAF image encodings with scalar context

The workflow implements rigorous stationarity testing (ADF & KPSS), feature engineering, model training, and evaluation using time-based train/validation/test splits to prevent data leakage.

---

## 🎯 Key Results

### 1-Step-Ahead Forecasting Performance

#### MLP Performance (lags=20)
- **Levels**: Strong performance (high persistence enables accurate 1-step forecasts)
- **Returns**: R² near 0 or negative (weak predictability due to noise dominance)
- **FracDiff (d=0.60)**: Moderate performance balancing stationarity and memory

#### CNN Performance (GAF + last_value, window=32)
- **Levels**: Strong performance after incorporating scalar context
- **Returns**: Remains challenging (near-zero predictability)
- **FracDiff**: Low but positive structure captured

> **💡 Key Insight**: The baseline CNN on GAF images alone failed on **levels** due to loss of absolute scale information. Adding the **last observed value** as scalar context successfully injects scale information, dramatically improving level forecasting.

---

## 📁 Project Structure

```text
.
├── notebooks/
│   ├── 01_step1_stationarity.ipynb    # Stationarity tests & transformations
│   ├── 02_step2_mlp.ipynb              # MLP lag-based forecasting
│   ├── 03_step3_cnn_gaf.ipynb          # CNN with GAF image encoding
│   └── 04_step4_compare.ipynb          # Model comparison & analysis
├── data/
│   └── step1_AAPL_1d_data.csv          # Processed AAPL time series data
├── results/
│   ├── step2_mlp_results.csv           # MLP evaluation metrics
│   ├── step3_cnn_gaf_scalar_results.csv # CNN evaluation metrics
│   └── step4_mlp_vs_cnn_comparison.csv  # Comparative analysis
├── figures/
│   ├── Fig01_price_levels.png          # Price visualization
│   ├── Fig02_hist_levels.png           # Distribution analysis
│   ├── ...
│   └── Fig33_r2_fracdiff_mlp_vs_cnn.png # Final comparison
├── report/
│   └── Project_Report.docx             # Comprehensive project report
├── TASKS.md                             # Project tasks and milestones
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

---

## 🚀 How to Run

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate

# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Execute Notebooks

Open VS Code with Jupyter extension and run notebooks **sequentially**:

1. `01_step1_stationarity.ipynb` — Data loading, stationarity tests, transformations
2. `02_step2_mlp.ipynb` — MLP model training and evaluation
3. `03_step3_cnn_gaf.ipynb` — CNN with GAF encoding training
4. `04_step4_compare.ipynb` — Comparative analysis and visualization

**Outputs**: All tables and metrics are saved to `results/`, figures can be exported to `figures/`.

---

## 🔬 Methodology

### Step 1: Stationarity & Transformations

#### Statistical Tests
- **ADF Test** (Augmented Dickey-Fuller): Tests the null hypothesis of a unit root (non-stationarity)
- **KPSS Test** (Kwiatkowski-Phillips-Schmidt-Shin): Tests the null hypothesis of stationarity

#### Transformations
- **Log Returns**:
  ```
  r_t = ln(P_t) - ln(P_{t-1})
  ```

- **Fractional Differencing** (fixed-width method):
  - Select smallest *d* satisfying ADF p-value < 0.05
  - Retain ≥ 1000 observations
  - Maximize correlation with original levels (preserve memory)

---

### Step 2: MLP Forecasting (Lag Features)

#### Architecture
- Supervised learning using lag vectors: `[y_{t-1}, ..., y_{t-p}]`
- Input features: 20 lagged observations (configurable)
- Hidden layers with ReLU activation
- Output: 1-step-ahead prediction

#### Training Protocol
- **Scaling**: StandardScaler fit on training set only (prevents leakage)
- **Optimizer**: Adam with configurable learning rate
- **Regularization**: Early stopping on validation loss
- **Loss**: Mean Squared Error (MSE)

---

### Step 3: CNN Forecasting (GAF Images + Scalar Context)

#### Image Encoding
- Convert sliding windows to **Gramian Angular Field (GAF)** images using summation method
- GAF captures temporal patterns as visual textures
- Window size: 32 time steps

#### Architecture
- **Convolutional layers** extract spatial-temporal features from GAF images
- **Scalar input**: Last observed value (`last_value`) concatenated before dense layers
- **Purpose of scalar context**: Preserves absolute scale information lost in GAF encoding

#### Training Protocol
- Early stopping on validation loss
- Adam optimizer
- Binary cross-entropy or MSE loss depending on task

---

### Step 4: Model Comparison

#### Evaluation Metrics
- **RMSE** (Root Mean Squared Error): Scale-dependent error measure
- **MAE** (Mean Absolute Error): Robust to outliers
- **R²** (Coefficient of Determination): Proportion of variance explained

#### Comparative Analysis
- Aggregate metrics across all three representations (Levels, Returns, FracDiff)
- Visualize performance using bar charts and line plots
- Statistical significance testing where applicable

---

## 📈 Figures Gallery

### Stationarity & Diagnostics
- Price levels with rolling statistics
- ACF/PACF plots for levels, returns, and fractional differences
- Histogram and distribution analysis
- Fractional differencing grid search results

### Model Predictions
- MLP predictions vs actual values (train/val/test)
- CNN (GAF+scalar) predictions vs actual values
- Training and validation loss curves
- Residual analysis plots

### Comparative Analysis
- MLP vs CNN performance across representations
- R² comparison bar charts
- RMSE/MAE heatmaps
- Model architecture diagrams

---

## 🔒 Reproducibility Notes

- **Chronological splits**: Train/validation/test splits maintain temporal order to prevent look-ahead bias
- **Train-only scaling**: Feature scaling and normalization use statistics from training set only
- **Random seeds**: TensorFlow and NumPy random seeds fixed where applicable
- **Version control**: All package versions specified in `requirements.txt`

---

## 📚 References

Aroussi, R. (n.d.). *yfinance* [Python package]. PyPI. https://pypi.org/project/yfinance/

Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. *Journal of the American Statistical Association, 74*(366a), 427–431. https://doi.org/10.1080/01621459.1979.10482531

Kingma, D. P., & Ba, J. (2014). *Adam: A method for stochastic optimization* (arXiv:1412.6980). https://arxiv.org/abs/1412.6980

Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. *Journal of Econometrics, 54*(1–3), 159–178. https://doi.org/10.1016/0304-4076(92)90104-Y

Lopez de Prado, M. (2018). *Advances in financial machine learning*. Wiley.

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature, 323*, 533–536. https://doi.org/10.1038/323533a0

Wang, Z., & Oates, T. (2015). Imaging time-series to improve classification and imputation. In *Proceedings of the 24th International Joint Conference on Artificial Intelligence* (IJCAI 2015) (pp. 3939–3945). AAAI Press.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is available under the MIT License. See LICENSE file for details.

---

## 📧 Contact

For questions or collaboration inquiries, please open an issue in the repository.

---

**⭐ If you find this project useful, please consider giving it a star!**