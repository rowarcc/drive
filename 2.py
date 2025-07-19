# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


def fit_sarimax(endog, exog=None, order=(1,1,0), seasonal_order=(1,0,0,12),
                train_end=None, test_start=None, test_end=None):
    """
    Fit SARIMAX on the training period and produce forecasts with 80% & 95% CI.
    
    Parameters:
    - endog: pd.Series, endogenous variable indexed by date.
    - exog: pd.DataFrame or None, exogenous variables aligned with endog.
    - order: tuple(p,d,q), ARIMA order.
    - seasonal_order: tuple(P,D,Q,s), seasonal order.
    - train_end: str or Timestamp, last month in training (inclusive, format 'YYYY-MM').
    - test_start: str or Timestamp, first month to forecast (format 'YYYY-MM').
    - test_end:   str or Timestamp, last month to forecast (format 'YYYY-MM').

    Returns:
    - results: fitted SARIMAXResults object.
    - train: pd.Series of training data.
    - test: pd.Series of actual test data (NaN for future).
    - forecast: pd.Series of forecasted mean.
    - ci80: pd.DataFrame of 80% confidence intervals.
    - ci95: pd.DataFrame of 95% confidence intervals.
    """
     
    # Create train and test indices
    train_end = pd.to_datetime(train_end)
    test_index = pd.date_range(start=test_start, end=test_end, freq='MS')
    
    train = endog.loc[:train_end]
    test = endog.reindex(test_index)
    
    exog_train = exog.loc[:train_end] if exog is not None else None
    exog_test = exog.reindex(test_index) if exog is not None else None
    
    # Fit model
    model = sm.tsa.statespace.SARIMAX(
        train,
        exog=exog_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    
    # Forecast
    steps = len(test_index)
    pred = results.get_forecast(steps=steps, exog=exog_test)
    forecast = pred.predicted_mean
    ci95 = pred.conf_int(alpha=0.05)
    ci80 = pred.conf_int(alpha=0.20)
    
    ci80 = ci80.rename(columns={
        ci80.columns[0]: 'CI80_lower',
        ci80.columns[1]: 'CI80_upper'
    })
    ci95 = ci95.rename(columns={
        ci95.columns[0]: 'CI95_lower',
        ci95.columns[1]: 'CI95_upper'
    })

    return results, train, test, forecast, ci80, ci95

def plot_forecast(train, test, forecast, ci80=None, ci95=None):
    """
    Plot training data, actual test data, and forecasts with optional confidence intervals.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(train, label='Train')
    plt.plot(test,  label='Actual')
    plt.plot(forecast, label='Forecast', linestyle='--')
    if ci95 is not None:
        plt.fill_between(ci95.index, ci95.iloc[:, 0], ci95.iloc[:, 1], alpha=0.15)
    if ci80 is not None:
        plt.fill_between(ci80.index, ci80.iloc[:, 0], ci80.iloc[:, 1], alpha=0.25)
    plt.legend()
    plt.show()

def evaluate_model(results, test, forecast):
    """
    Compute and display key metrics:
    
    - AIC (Akaike Information Criterion): lower is better; balances fit vs complexity.
    - BIC (Bayesian Information Criterion): lower is better; stronger penalty for complexity.
    - AICc (Corrected AIC): adjusts AIC for small samples; lower is better.
    - RMSE (Root Mean Squared Error): average magnitude of errors; lower indicates more accurate forecasts.
    - MAPE (Mean Absolute Percentage Error): average % error; lower (<10-20%) indicates good predictive accuracy.
    """
    k = len(results.params)
    n = results.nobs
    aic = results.aic
    bic = results.bic
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    
    # Compare only where actuals exist
    mask = test.notna()
    te = test[mask]
    fe = forecast[mask]
    
    rmse = np.sqrt(((fe - te) ** 2).mean())
    mape = (np.abs((fe - te) / te).mean()) * 100
    
    print(f"AIC:  {aic:.3f}")
    print(f"AICc: {aicc:.3f}")
    print(f"BIC:  {bic:.3f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")


# -------------------------------
# 1) Load and prepare raw data
# -------------------------------
df = pd.read_csv('data.txt', sep='|', parse_dates=['elig_month'])
df.set_index('elig_month', inplace=True)

# Create new features
df['plan_proportion_DCSNP'] = df['plan_proportion_CSNP'] + df['plan_proportion_DSNP']
df['z_hcc_factor']    = (df['hcc_factor']   - df['hcc_factor'].mean())   / df['hcc_factor'].std()
df['z_member_count']  = (df['member_count'] - df['member_count'].mean()) / df['member_count'].std()
df['v28']             = (df.index > '2024-01-31').astype(int)

# Define endogenous and initial exogenous
y    = df['raf_restate']
exog = df[['plan_proportion_DCSNP','plan_proportion_MA','z_hcc_factor','z_member_count','v28']]

# -------------------------------
# 2) Handle missing exog for future months
# -------------------------------
# Define forecasting scenarios
scenarios = [
    ('2023-12','2024-01','2025-06'),
    ('2024-05','2024-06','2025-06'),
    ('2025-01','2025-02','2025-12')
]
# Compute latest month needed
max_end = max(pd.to_datetime(end) for _,_,end in scenarios)

# Build a full monthly index from first data month to max_end
full_index = pd.date_range(start=df.index.min(), end=max_end, freq='MS')

# Reindex exog to full horizon
exog_full = exog.reindex(full_index)

# Fill missing by copying same month last year
exog_full = exog_full.fillna(exog_full.shift(12))

# Example manual overrides if needed:
# exog_full.loc[pd.to_datetime('2025-07'),'plan_proportion_MA'] = 0.30
# exog_full.loc[pd.to_datetime('2025-08'),'plan_proportion_MA'] = 0.32
# exog_full.loc[pd.to_datetime('2025-07'),'v28'] = 1

# -------------------------------
# 4) Loop through scenarios
# -------------------------------
for train_end, test_start, test_end in scenarios:
    print(f"\n--- Scenario: train={train_end}, test={test_start}→{test_end} ---")
    res, tr, te, fc, ci80, ci95 = fit_sarimax(
        y, exog_full, train_end=train_end,
        test_start=test_start, test_end=test_end
    )
    results = pd.concat([
        tr.rename('train'),
        te.rename('actual'),
        fc.rename('forecast'),
        ci80,
        ci95
    ], axis=1)
    evaluate_model(res, te, fc)
    plot_forecast(tr, te, fc, ci80, ci95)

# %%
pd.set_option('display.max_rows', None)
print(results)
