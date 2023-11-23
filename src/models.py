from src.data_preprocessing import load_data
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pandas as pd


def train_arima_model(time_series, arima_order):
    model = ARIMA(time_series, order=arima_order)
    model_fit = model.fit()
    return model_fit


def forecast_with_model(model_fit, steps):
    forecast = model_fit.forecast(steps=steps)
    return forecast


def run_model_pipeline():
    # Load data
    time_series_path = "../data/processed/processed_data.csv"
    train_data = load_data(time_series_path)

    # Convert 'order_date' to datetime and set it as index if not already done in load_data
    train_data['order_date'] = pd.to_datetime(train_data['order_date'])
    train_data.set_index('order_date', inplace=True)

    # # Resample the data to a daily frequency, filling missing days with NaN or an interpolation
    # train_data = train_data.resample('D').mean().interpolate()

    # Assume 'ord_qty' is the column you want to forecast
    time_series = train_data['ord_qty']

    # Train the model
    arima_order = (5, 1, 0)  # Example order, tune this based on your data
    model_fit = train_arima_model(time_series, arima_order)

    # Forecast
    forecast_steps = 90  # Example for 3 months, adjust based on your data
    forecast = forecast_with_model(model_fit, forecast_steps)

    # Plot the forecast against the actual data
    plt.figure(figsize=(12, 7))
    plt.plot(time_series, label='Actual', alpha=0.5)
    plt.plot(time_series.index[-forecast_steps:], forecast, label='Forecast')
    plt.title('ARIMA Model Forecast vs Actuals')
    plt.legend(loc='best')
    plt.show()

    # Model Diagnostics
    residuals = model_fit.resid
    plt.figure(figsize=(12, 7))
    residuals.plot(title="Residuals")
    plt.show()

    # ACF and PACF of residuals
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_acf(residuals, ax=axes[0], title="ACF of Residuals")
    plot_pacf(residuals, ax=axes[1], title="PACF of Residuals")
    plt.show()

    # Additional diagnostic plots
    try:
        model_fit.plot_diagnostics(figsize=(15, 12))
        plt.show()
    except AttributeError:
        print("plot_diagnostics method is not available.")


if __name__ == "__main__":
    run_model_pipeline()
