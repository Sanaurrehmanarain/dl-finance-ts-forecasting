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
├── imgs/                               # Directory for images
├── .gitignore                          # Git ignore rules
├── dl1.png                             # Project banner image
├── main.ipynb                          # Main Jupyter Notebook containing all code and experiments
├── main.pdf                            # PDF export of the Jupyter Notebook
├── Project_tasks.pdf                   # Project requirements and tasks
├── README.md                           # This documentation file
├── Report.pdf                          # Comprehensive project report (PDF format)
├── step1_AAPL_1d_data.csv              # Processed AAPL time series data
├── step2_mlp_results.csv               # Evaluation metrics for the MLP models
├── step3_cnn_gaf_results.csv           # Evaluation metrics for the baseline CNN models
├── step3_cnn_gaf_scalar_results.csv    # Evaluation metrics for the CNN (GAF + scalar) models
├── step4_mlp_vs_cnn_comparison.csv     # Final comparative analysis metrics
└── TASKS.md                            # Detailed project milestones
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

Open VS Code with the Jupyter extension and run the cells in main.ipynb sequentially. The notebook encompasses:

1. Data loading, stationarity tests, and transformations

2. MLP model training and evaluation

3. CNN with GAF encoding training

4. Comparative analysis and visualization

All output tables and metrics will be generated as .csv files in the root directory.

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

## 📈 Results Visualization

1. The Data: Non-Stationary Price Levels
Raw AAPL price levels exhibit strong persistence and upward trends, highlighting the need for stationarity testing.
(Image: imgs/price_levels.png)

2. Feature Engineering: Gramian Angular Field (GAF)
Time-series windows are encoded into 2D images to allow Convolutional Neural Networks to extract spatial-temporal textures.
(Image: imgs/gaf_sample.png)

3. Forecasting: MLP Prediction on Levels
The lag-based Multi-Layer Perceptron successfully captures the 1-step-ahead persistence of the raw price levels.
(Image: imgs/mlp_prediction.png)

4. Final Model Comparison
Comparing Test RMSE across the three data representations for both MLP and CNN architectures.
(Image: imgs/final_comparison.png)

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

## Citation

If you use this project in academic research, publications, educational
materials, or derivative works, please cite the project.

This repository includes a CITATION.cff file, so GitHub provides a
"Cite this repository" button in the repository sidebar. You can use it
to obtain citations in BibTeX, APA, and other supported formats.

Suggested citation:

Arain, S. U. R. (2026). [dl-finance-ts-forecasting] (Version 1.0) [Software].
https://github.com/sanaurrehmanarain/dl-finance-ts-forecasting

Author: Sana Ur Rehman Arain

Profession: Data Scientist

GitHub: https://github.com/sanaurrehmanarain

Contact: sana.arain.work@gmail.com

If you build upon this work, attribution is appreciated and helps others
discover the original project.

Note: The MIT License requires that the original copyright
notice be retained in copies of the Software.

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

**⭐ If you find this project useful, please consider giving it a star!**
