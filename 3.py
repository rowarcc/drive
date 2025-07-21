import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -----------------------------------------------------------------------------
# 1. Plotting function
# -----------------------------------------------------------------------------
def plot_forecast(results_df, title):
    """
    Plot train, actual, forecast and 80/95% CIs from a unified results DataFrame.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must have columns ['train','actual','forecast',
        'ci80lower','ci80upper','ci95lower','ci95upper'], indexed by date.
    title : str
        Title for the chart.
    """
    plt.figure(figsize=(10, 6))
    # plot train, actual, forecast if present
    for col, style in zip(['train','actual','forecast'], ['-', '-', '--']):
        if col in results_df:
            results_df[col].plot(label=col, linestyle=style)
    # fill confidence bands
    idx = results_df.index
    plt.fill_between(idx,
                     results_df['ci80lower'],
                     results_df['ci80upper'],
                     alpha=0.3,
                     label='80% CI')
    plt.fill_between(idx,
                     results_df['ci95lower'],
                     results_df['ci95upper'],
                     alpha=0.2,
                     label='95% CI')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# 2. Forecast‐evaluation (ETS & SARIMAX share this)
# -----------------------------------------------------------------------------
def evaluate_forecast(actual, forecast):
    """
    Compute MAE, RMSE, MAPE on overlapping (non‐NaN) periods.
    Returns a dict.
    """
    df = pd.concat([actual, forecast], axis=1).dropna()
    df.columns = ['actual','forecast']
    mae  = np.mean(np.abs(df['actual'] - df['forecast']))
    rmse = np.sqrt(np.mean((df['actual'] - df['forecast'])**2))
    mape = np.mean(np.abs((df['actual'] - df['forecast']) / df['actual'])) * 100
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


# -----------------------------------------------------------------------------
# 3. SARIMAX‐evaluation (AIC, AICc, BIC)
# -----------------------------------------------------------------------------
def evaluate_sarimax(fit_result, n_obs):
    """
    Given a fitted SARIMAXResults and number of observations (n_obs),
    return AIC, AICc, BIC as a dict.
    """
    aic = fit_result.aic
    bic = fit_result.bic
    k = len(fit_result.params)
    # corrected AIC
    aicc = aic + (2 * k * (k + 1)) / (n_obs - k - 1)
    return {'AIC': aic, 'AICc': aicc, 'BIC': bic}


# -----------------------------------------------------------------------------
# 4. ETS wrapper
# -----------------------------------------------------------------------------
def damped_hw_ets(endog, train_end, test_start, test_end,
                  seasonal_periods=12, trend='add', seasonal='add', damped=True):
    """
    Slice endog by train_end/test_start/test_end, fit damped Holt–Winters,
    and return a DataFrame with train, actual, forecast, ci80/95.
    """
    # ensure timestamps
    train_end = pd.to_datetime(train_end)
    test_idx  = pd.date_range(start=test_start, end=test_end, freq='MS')

    # split
    train = endog.loc[:train_end]
    test  = endog.reindex(test_idx)

    # fit & forecast
    model = ExponentialSmoothing(train,
                                 trend=trend,
                                 damped_trend=damped,
                                 seasonal=seasonal,
                                 seasonal_periods=seasonal_periods)
    fit = model.fit()
    h = len(test)
    fc = fit.forecast(h).reindex(test_idx)

    # compute resid‐based sigma
    sigma = (train - fit.fittedvalues).std()
    steps = np.arange(1, h+1)
    se = sigma * np.sqrt(steps)
    z80, z95 = 1.28155, 1.96

    ci80l = fc - z80 * se
    ci80u = fc + z80 * se
    ci95l = fc - z95 * se
    ci95u = fc + z95 * se

    # assemble full results
    full_idx = train.index.union(test_idx)
    results = pd.DataFrame(index=full_idx)
    results['train']       = train
    results['actual']      = results['train'].where(results.index <= train_end).append(test)
    results['forecast']    = fc.reindex(full_idx)
    results['ci80lower']   = ci80l.reindex(full_idx)
    results['ci80upper']   = ci80u.reindex(full_idx)
    results['ci95lower']   = ci95l.reindex(full_idx)
    results['ci95upper']   = ci95u.reindex(full_idx)

    return results


# -----------------------------------------------------------------------------
# 5. SARIMAX wrapper
# -----------------------------------------------------------------------------
def sarimax_model(endog, exog, train_end, test_start, test_end,
                  order=(1,1,0), seasonal_order=(1,0,0,12)):
    """
    Slice endog & exog by train/test dates, fill exog‐gaps by same month last year,
    fit SARIMAX, forecast, and return (fit_result, results_df).
    """
    train_end = pd.to_datetime(train_end)
    test_idx  = pd.date_range(start=test_start, end=test_end, freq='MS')

    # split endog
    train = endog.loc[:train_end]
    test  = endog.reindex(test_idx)

    # prepare exog: extend to cover test_idx, then fill missing via 12-month lag
    full_exog = exog.reindex(exog.index.union(test_idx))
    full_exog = full_exog.fillna(full_exog.shift(12))
    exog_train = full_exog.loc[:train_end]
    exog_test  = full_exog.loc[test_idx]

    # fit
    mod = sm.tsa.statespace.SARIMAX(train,
                                    exog=exog_train,
                                    order=order,
                                    seasonal_order=seasonal_order,
                                    enforce_stationarity=False,
                                    enforce_invertibility=False)
    res = mod.fit(disp=False)

    # forecast + CIs
    pred = res.get_forecast(steps=len(test_idx), exog=exog_test)
    fc   = pred.predicted_mean
    ci95 = pred.conf_int(alpha=0.05)
    ci80 = pred.conf_int(alpha=0.20)

    # rename
    ci80.columns = ['ci80lower','ci80upper']
    ci95.columns = ['ci95lower','ci95upper']

    # assemble full results
    full_idx = train.index.union(test_idx)
    results = pd.DataFrame(index=full_idx)
    results['train']     = train
    results['actual']    = results['train'].where(results.index <= train_end).append(test)
    results['forecast']  = fc.reindex(full_idx)
    results = results.join(ci80.reindex(full_idx)).join(ci95.reindex(full_idx))

    return res, results


# -----------------------------------------------------------------------------
# EXAMPLE USAGE
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    # --- Load & prepare ---
    df = pd.read_csv('data.txt', sep='|', parse_dates=['elig_month'])
    df.set_index('elig_month', inplace=True)
    df = df.asfreq('MS')

    y = df['raf_restate']
    # build exog and fill future gaps once for all
    exog = df[['plan_proportion_CSNP',
               'plan_proportion_DSNP',
               'plan_proportion_MA',
               'hcc_factor',
               'member_count']]
    exog_full = exog.copy()
    # example: combine CSNP+DSNP into DCSNP
    exog_full['plan_proportion_DCSNP'] = exog_full['plan_proportion_CSNP'] + exog_full['plan_proportion_DSNP']
    exog_full['z_hcc_factor']   = (exog_full['hcc_factor']   - exog_full['hcc_factor'].mean())   / exog_full['hcc_factor'].std()
    exog_full['z_member_count'] = (exog_full['member_count'] - exog_full['member_count'].mean()) / exog_full['member_count'].std()
    exog_full = exog_full[['plan_proportion_DCSNP','plan_proportion_MA','z_hcc_factor','z_member_count']]

    # your scenarios
    scenarios = [
        ('2023-01-01','2024-01-01','2025-06-01'),
        ('2023-01-01','2024-05-01','2025-06-01'),
        ('2023-01-01','2024-01-01','2025-07-01'),
    ]

    for train_end, test_start, test_end in scenarios:
        print(f"\n=== ETS scenario {train_end} → {test_start}–{test_end} ===")
        ets_res = damped_hw_ets(y, train_end, test_start, test_end)
        plot_forecast(ets_res, 'ETS: raf_restate')
        ets_metrics = evaluate_forecast(ets_res['actual'], ets_res['forecast'])
        print("ETS metrics:", ets_metrics)

        print(f"\n--- SARIMAX scenario {train_end} → {test_start}–{test_end} ---")
        sarimax_fit, sarimax_res = sarimax_model(y, exog_full, train_end, test_start, test_end)
        plot_forecast(sarimax_res, 'SARIMAX: raf_restate')
        sarimax_metrics = evaluate_sarimax(sarimax_fit, sarimax_fit.nobs)
        print("SARIMAX info criteria:", sarimax_metrics)
        offset_metrics = evaluate_forecast(sarimax_res['actual'], sarimax_res['forecast'])
        print("SARIMAX forecast errors:", offset_metrics)
