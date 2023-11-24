import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def describe_data(data):
    """
    Display descriptive statistics of the given DataFrame.

    Parameters:
    data (pandas.DataFrame): The DataFrame to describe.
    """
    print("Descriptive Statistics:\n", data.describe())


def plot_histograms(data):
    """
    Plot histograms for each numerical column in the DataFrame.

    Parameters:
    data (pandas.DataFrame): The DataFrame for which to plot histograms.
    """
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(data[column], kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()


def plot_time_trend(data):
    """
    Plot time trend if 'order_date' is present in the DataFrame.

    Parameters:
    data (pandas.DataFrame): The DataFrame to analyze.
    """
    if 'order_date' in data.columns:
        plt.figure(figsize=(10, 5))
        data.groupby('order_date').size().plot()
        plt.title('Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Orders')
        plt.show()


def analyze_time_series(data):
    """
    Perform time series analysis, including trends and seasonality analysis.

    Parameters:
    data (pandas.DataFrame): The DataFrame on which to perform time series analysis.
    """
    if 'order_date' in data.columns:
        data['order_date'] = pd.to_datetime(data['order_date'])
        data.set_index('order_date', inplace=True)

        monthly_ord_qty = data['ord_qty'].resample('M').mean()

        plt.figure(figsize=(12, 6))
        monthly_ord_qty.plot(title='Monthly Average Order Quantity', color='blue', marker='o')
        plt.xlabel('Month')
        plt.ylabel('Average Order Quantity')
        plt.show()


def perform_eda(data):
    """
    Perform a series of exploratory data analysis on the given DataFrame.

    Parameters:
    data (pandas.DataFrame): The DataFrame on which to perform EDA.
    """
    describe_data(data)
    plot_histograms(data)
    plot_time_trend(data)
    analyze_time_series(data)


def main():
    file_path = "../data/processed/processed_data.csv"
    data = pd.read_csv(file_path)
    perform_eda(data)


if __name__ == "__main__":
    main()
