# Money Modeling

This project contains a Jupyter notebook (`modeling.ipynb`) that demonstrates Random Forest regression modeling using scikit-learn. The notebook includes:

- Data preprocessing and feature engineering
- Training a Random Forest regressor
- Model evaluation (MAE, MSE, RMSE)
- Feature importance analysis

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

1. Open the notebook:
   ```bash
   jupyter notebook modeling.ipynb
   ```
2. Run the cells to train and evaluate the model.

## Dependencies
- scikit-learn
- numpy
- pandas
- matplotlib 

## Statistical Tests & Diagnostics

This project applies several statistical tests and diagnostics to ensure the robustness and validity of financial time series modeling. Below is an overview of each test, its purpose, and its significance in finance:

---

### 1. Augmented Dickey-Fuller (ADF) Test
- **Purpose:** Tests for stationarity in a time series (i.e., whether statistical properties like mean and variance are constant over time).
- **How it works:** The null hypothesis is that the series has a unit root (is non-stationary). A low p-value (< 0.05) suggests the series is stationary.
- **Why it matters in finance:** Many financial models (e.g., ARIMA) require stationary data. Non-stationary price series can lead to spurious results and unreliable forecasts.
- **Usage in notebook:** Used to determine if differencing is needed before ARIMA modeling.

**Example:**
```python
from statsmodels.tsa.stattools import adfuller
result = adfuller(price_series)
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
```
*If p-value < 0.05, the series is stationary.*

---

### 2. KPSS (Kwiatkowski-Phillips-Schmidt-Shin) Test
- **Purpose:** Also tests for stationarity, but with the opposite null hypothesis (that the series is stationary).
- **How it works:** A low p-value (< 0.05) suggests the series is non-stationary.
- **Why it matters in finance:** Used in conjunction with ADF to robustly assess stationarity, reducing the risk of misclassification.
- **Usage in notebook:** Confirms the result of the ADF test for optimal differencing.

**Example:**
```python
from statsmodels.tsa.stattools import kpss
stat, p_value, _, _ = kpss(price_series, regression='c')
print(f"KPSS Statistic: {stat:.4f}")
print(f"p-value: {p_value:.4f}")
```
*If p-value < 0.05, the series is non-stationary.*

---

### 3. Ljung-Box Test
- **Purpose:** Checks for autocorrelation in the residuals of a fitted time series model.
- **How it works:** The null hypothesis is that residuals are independently distributed. A high p-value (> 0.05) means no significant autocorrelation remains.
- **Why it matters in finance:** Ensures that the model has captured all predictable structure; residual autocorrelation indicates model inadequacy.
- **Usage in notebook:** Diagnostic for ARIMA model residuals.

**Example:**
```python
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(lb_test)
```
*Look for p-value > 0.05 for no autocorrelation in residuals.*

---

### 4. Jarque-Bera Test
- **Purpose:** Tests whether residuals are normally distributed.
- **How it works:** The null hypothesis is that the data is normally distributed. A high p-value (> 0.05) supports normality.
- **Why it matters in finance:** Many statistical inference techniques assume normality of errors. Non-normal residuals can indicate model misspecification or the presence of outliers.
- **Usage in notebook:** Diagnostic for ARIMA model residuals.

**Example:**
```python
from scipy.stats import jarque_bera
jb_stat, jb_pvalue = jarque_bera(residuals)
print(f"Jarque-Bera p-value: {jb_pvalue:.4f}")
```
*Look for p-value > 0.05 for normal residuals.*

---

### 5. Model Evaluation Metrics
| Metric | Description | Financial Significance |
|--------|-------------|-----------------------|
| **MAE** (Mean Absolute Error) | Average absolute difference between predicted and actual values | Measures average prediction error in the same units as the data |
| **MSE** (Mean Squared Error) | Average squared difference between predicted and actual values | Penalizes larger errors more heavily; sensitive to outliers |
| **RMSE** (Root Mean Squared Error) | Square root of MSE | Interpretable in original units; commonly used in finance |
| **R²** (Coefficient of Determination) | Proportion of variance explained by the model | Indicates goodness-of-fit; higher is better |

**Example:**
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_true, y_pred)
print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}")
```

---

### 6. Autocorrelation & Partial Autocorrelation (ACF/PACF)
- **Purpose:** Visual tools to identify the presence and order of autocorrelation in time series data.
- **Why it matters in finance:** Helps in selecting appropriate ARIMA model parameters (p, q) and diagnosing model fit.

**Example:**
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
plot_acf(price_series, lags=40)
plt.title('Autocorrelation Function (ACF)')
plt.show()
plot_pacf(price_series, lags=40)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
```
*Look for significant spikes to determine AR (PACF) and MA (ACF) orders.*

---

### 7. Rolling Mean & Standard Deviation
- **Purpose:** Detects structural breaks, regime shifts, or volatility changes in financial time series.
- **Why it matters in finance:** Financial markets often experience changing volatility and mean shifts, which can impact model performance and risk assessment.

**Example:**
```python
import pandas as pd
import matplotlib.pyplot as plt
rolling_mean = pd.Series(price_series).rolling(window=50).mean()
rolling_std = pd.Series(price_series).rolling(window=50).std()
plt.plot(price_series, label='Price')
plt.plot(rolling_mean, label='Rolling Mean', color='red')
plt.plot(rolling_std, label='Rolling Std', color='orange')
plt.legend()
plt.title('Rolling Mean & Std Deviation')
plt.show()
```
*Look for shifts in mean or volatility as indicators of regime change or instability.*

---

**In summary:**
These statistical tests and diagnostics are essential for building reliable, interpretable, and robust financial models. They help ensure that the assumptions underlying time series models are met, and that the results are meaningful for financial decision-making.

## Neural Network Techniques

This project explores several advanced neural network architectures for financial time series forecasting. Below are the main techniques used, with explanations of their structure, application, and relevance to finance:

---

### 1. Long Short-Term Memory (LSTM) Networks
- **What:** LSTM is a type of recurrent neural network (RNN) designed to capture long-term dependencies in sequential data by using memory cells and gating mechanisms.
- **Usage in notebook:** Used as a baseline neural network for predicting stock prices based on historical sequences of features.
- **Strengths:** Good at modeling temporal dependencies and handling vanishing gradient problems common in standard RNNs.
- **Weaknesses:** Can be slow to train and may overfit on small datasets.
- **Financial relevance:** Well-suited for capturing patterns in financial time series, such as trends and cycles.

---

### 2. Gated Recurrent Unit (GRU) Networks
- **What:** GRU is a simplified variant of LSTM that uses fewer gates, making it computationally more efficient while retaining the ability to model sequential dependencies.
- **Usage in notebook:** Implemented as an alternative to LSTM for stock price prediction.
- **Strengths:** Faster training and often similar performance to LSTM; less prone to overfitting on small data.
- **Weaknesses:** May be less expressive than LSTM for very complex patterns.
- **Financial relevance:** Effective for time series with moderate complexity and limited data.

---

### 3. Bidirectional GRU
- **What:** A GRU network that processes input sequences in both forward and backward directions, allowing the model to learn from both past and future context.
- **Usage in notebook:** Used to enhance the model's ability to capture context in financial sequences.
- **Strengths:** Can improve accuracy by leveraging information from the entire sequence.
- **Weaknesses:** More computationally intensive; not always suitable for real-time prediction.
- **Financial relevance:** Useful for retrospective analysis and scenarios where future context is available.

---

### 4. CNN-LSTM Hybrid
- **What:** Combines 1D convolutional layers (CNN) for feature extraction with LSTM layers for sequence modeling.
- **Usage in notebook:** Applied to capture both local patterns (via CNN) and long-term dependencies (via LSTM) in financial data.
- **Strengths:** Can extract complex features and temporal relationships; often improves performance on noisy data.
- **Weaknesses:** More complex architecture; requires careful tuning.
- **Financial relevance:** Suitable for financial data with both short-term and long-term patterns (e.g., price spikes, trends).

---

### 5. Deep/Stacked LSTM & GRU Architectures
- **What:** Multiple LSTM or GRU layers stacked to increase model capacity and capture hierarchical temporal features.
- **Usage in notebook:** Used to build more expressive models for challenging prediction tasks.
- **Strengths:** Can model complex, multi-scale temporal dependencies.
- **Weaknesses:** Higher risk of overfitting; longer training times.
- **Financial relevance:** Useful for modeling intricate market dynamics and multi-factor influences.

---

### 6. Ensemble Neural Networks
- **What:** Combines predictions from multiple neural network models (e.g., LSTM, GRU, CNN-LSTM) to improve robustness and accuracy.
- **Usage in notebook:** Ensemble predictions are compared to individual model results for performance benchmarking.
- **Strengths:** Reduces model variance and leverages strengths of different architectures.
- **Weaknesses:** Increases computational cost and complexity.
- **Financial relevance:** Helps mitigate overfitting and improves generalization in volatile markets.

---

**In summary:**
Neural network techniques provide powerful tools for modeling complex, nonlinear relationships in financial time series. This project demonstrates and compares several architectures, highlighting their strengths and trade-offs for stock price prediction. 