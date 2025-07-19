# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def fit_damped_hw(train, test,
                  seasonal_periods=12,
                  trend='add',
                  seasonal='add',
                  damped=True):
    """
    Fit a damped Holt–Winters model on `train`, then forecast the same length as `test`.

    Parameters
    ----------
    train : pd.Series
        Training data, indexed by datetime.
    test : pd.Series
        Test/index series; its index (including future dates beyond available data)
        determines the forecast horizon.

    Returns
    -------
    forecast : pd.Series
        The point forecasts, indexed the same as `test`.
    ci_df : pd.DataFrame
        A DataFrame with columns ['lower_80','upper_80','lower_95','upper_95'],
        indexed the same as `test`.
    """
    # 1) Fit model
    model = ExponentialSmoothing(
        train,
        trend=trend,
        damped_trend=damped,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    )
    fit = model.fit()

    # 2) Forecast out for every period in `test`
    h = len(test)
    forecast = fit.forecast(h)
    forecast.index = test.index

    # 3) Estimate residual σ from training fit
    resid = train - fit.fittedvalues
    sigma = resid.std()

    # 4) Compute standard errors: assume se_h = σ * sqrt(h_step)
    h_steps = np.arange(1, h+1)
    se = sigma * np.sqrt(h_steps)

    # 5) Build 80% & 95% intervals
    z80, z95 = 1.28155, 1.96
    lower_80 = forecast - z80 * se
    upper_80 = forecast + z80 * se
    lower_95 = forecast - z95 * se
    upper_95 = forecast + z95 * se

    ci_df = pd.DataFrame({
        'lower_80': lower_80,
        'upper_80': upper_80,
        'lower_95': lower_95,
        'upper_95': upper_95
    }, index=test.index)

    return forecast, ci_df


def plot_forecast(results_df, value_name):
    """
    Plot train, actual, forecast, and 80%/95% CIs.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns: ['train','actual','forecast',
        'lower_80','upper_80','lower_95','upper_95'].
    value_name : str
        Name of the series (for axis label & title).
    """
    plt.figure(figsize=(10, 6))
    # plot each
    for col in ['train', 'actual', 'forecast']:
        if col in results_df:
            results_df[col].plot(label=col)
    # fill intervals
    plt.fill_between(results_df.index,
                     results_df['lower_80'],
                     results_df['upper_80'],
                     alpha=0.3,
                     label='80% CI')
    plt.fill_between(results_df.index,
                     results_df['lower_95'],
                     results_df['upper_95'],
                     alpha=0.2,
                     label='95% CI')

    plt.title(f'Damped Holt–Winters Forecast: {value_name}')
    plt.xlabel('Date')
    plt.ylabel(value_name)
    plt.legend()
    plt.tight_layout()
    plt.show()


def evaluate_forecast(actual, forecast):
    """
    Compute MAE, RMSE, MAPE on overlapping (non‐NaN) periods.

    Metrics interpretation:
      - MAE (mean absolute error): average size of errors—lower is better.
      - RMSE (root mean squared error): penalizes large errors more heavily—lower is better.
      - MAPE (mean absolute % error): error relative to magnitude. 
         • <10% = highly accurate 
         • 10–20% = good 
         • 20–50% = reasonable
    """
    df = pd.concat([actual, forecast], axis=1).dropna()
    df.columns = ['actual', 'forecast']

    mae  = np.mean(np.abs(df['actual'] - df['forecast']))
    rmse = np.sqrt(np.mean((df['actual'] - df['forecast'])**2))
    mape = np.mean(np.abs((df['actual'] - df['forecast']) / df['actual'])) * 100

    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


if __name__ == '__main__':
    # --- Load & prepare ---
    df = pd.read_csv('data.txt', sep='|', parse_dates=['elig_month'])
    df.set_index('elig_month', inplace=True)
    df = df.asfreq('MS')               # ensure monthly start frequency
    ts = df['raf_restate']

    # --- Define train/test scenarios ---
    # (train_end, test_start, test_end)
    scenarios = [
        ('2023-12', '2024-01', '2025-06'),   # long horizon (18 mo)
        ('2024-05', '2024-06', '2025-06'),   # very short horizon (1 mo)
        ('2025-02', '2025-03', '2025-12')    # medium horizon (10 mo; extends past data)
    ]

    for train_end, test_start, test_end in scenarios:
        print(f'\nScenario: train → {train_end}, test {test_start} to {test_end}')
        # slice train
        train = ts[:train_end]
        # build test index (even beyond available data)
        test_idx = pd.date_range(start=test_start, end=test_end, freq='MS')
        test    = ts.reindex(test_idx)

        # 1) Model & forecast
        forecast, ci = fit_damped_hw(train, test)

        # 2) Combine for plotting
        results = pd.concat([
            train.rename('train'),
            test.rename('actual'),
            forecast.rename('forecast'),
            ci
        ], axis=1)

        # 3) Plot
        plot_forecast(results, 'raf_restate')

        # 4) Evaluate
        metrics = evaluate_forecast(results['actual'], results['forecast'])
        print('Evaluation:')
        for name, val in metrics.items():
            print(f'  {name}: {val:.3f}')
