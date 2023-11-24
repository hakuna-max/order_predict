import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Load data from a CSV file.
    
    Parameters:
    file_path (str): The path to the CSV file to be loaded.

    Returns:
    pandas.DataFrame: Loaded data as a pandas DataFrame.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"file {file_path} does not found。")
        return None
    except Exception as e:
        print(f"Loading data error: {e}")
        return None
    

def preprocess_data(data):
    """
    Preprocess the given DataFrame, including date conversion and encoding categorical variables.

    Parameters:
    data (pandas.DataFrame): The DataFrame to preprocess.

    Returns:
    pandas.DataFrame: The preprocessed DataFrame.
    """
    # Convert date format
    if 'order_date' in data.columns:
        data['order_date'] = pd.to_datetime(data['order_date'])

    # Encode categorical variables
    if 'sales_chan_name' in data.columns:
        data['sales_chan_name'] = data['sales_chan_name'].map({'offline': 0, 'online': 1})
    
    # Handle outliers for numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        data[col] = handle_outliers(data[col])

    return data


def handle_outliers(series):
    """
    Handle outliers in a pandas series using the Interquartile Range (IQR) method.

    Parameters:
    series (pandas.Series): The pandas series in which to handle outliers.

    Returns:
    pandas.Series: The series with outliers handled.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Capping the outliers
    series = np.where(series < lower_bound, lower_bound, series)
    series = np.where(series > upper_bound, upper_bound, series)
    
    return series


def main():
    """
    Main function to run the data preprocessing steps.
    """
    # Example file path (modify as needed)
    file_path = "../data/raw/order_train0.csv"

    # Load data
    data = load_data(file_path)

    # Preprocess data
    preprocessed_data = preprocess_data(data)
    print("preprocessing raw data is done!")

    # Optionally, save or return the preprocessed data
    preprocessed_data.to_csv('../data/processed/processed_data.csv', index=False)
    print("Preprocessed data is saved in folder data/processed")
    return preprocessed_data


if __name__ == "__main__":
    main()