import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import holidays


def extract_date_features(data):
    """
    Extract features from the date.
    :param data: pandas DataFrame
    :return: DataFrame with added date features
    """
    data["year"] = data["order_date"].dt.year
    data["month"] = data["order_date"].dt.month
    data["day"] = data["order_date"].dt.day
    data["weekday"] = data["order_date"].dt.weekday
    # Assign seasons based on the Northern Hemisphere
    data["season"] = data["month"].apply(lambda x: (x % 12 + 3) // 3)
    return data


def encode_categorical_features(data):
    """
    Encode categorical features.
    :param data: pandas DataFrame
    :return: Encoded DataFrame
    """
    # One-hot encoding example - encoding 'sales_chan_name'
    encoder = OneHotEncoder(sparse_output=False)
    sales_chan_encoded = encoder.fit_transform(data[["sales_chan_name"]])
    sales_chan_encoded_df = pd.DataFrame(
        sales_chan_encoded, columns=encoder.get_feature_names_out(["sales_chan_name"])
    )
    data = pd.concat([data, sales_chan_encoded_df], axis=1).drop(
        "sales_chan_name", axis=1
    )
    return data


def encode_all_categorical_features(data, date_column):
    """
    Encode all categorical features in the DataFrame.
    :param data: pandas DataFrame
    :param date_column: The name of the date column, which should not be encoded.
    :return: DataFrame with encoded categorical features
    """
    categorical_columns = data.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        if col != date_column:  # Assuming date_column is the name of the date column
            encoder = OneHotEncoder(sparse_output=False)
            encoded = encoder.fit_transform(data[[col]])
            encoded_df = pd.DataFrame(
                encoded, columns=encoder.get_feature_names_out([col])
            )
            data = pd.concat([data, encoded_df], axis=1).drop(col, axis=1)
    return data


def add_season_feature(data):
    """
    Add a season feature, assuming Northern Hemisphere seasons.
    :param data: pandas DataFrame
    :return: DataFrame with added season feature
    """

    def get_season(month):
        if month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        elif month in [9, 10, 11]:
            return "Fall"
        else:
            return "Winter"

    data["season"] = data["month"].apply(get_season)
    return data


def add_holiday_feature(data, years):
    """
    Add a holiday feature for the dataset.
    :param data: pandas DataFrame
    :param years: list of years
    :return: DataFrame with holiday feature
    """
    cn_holidays = holidays.CN(years=years)
    data["is_holiday"] = data["order_date"].apply(lambda x: x in cn_holidays)
    return data


def add_month_phase_feature(data, date_column):
    """
    Add a feature for early, mid, or end of the month.
    :param data: pandas DataFrame
    :param date_column: name of the date column for extracting month phase
    :return: DataFrame
    """

    def categorize_month_phase(day):
        if day <= 10:
            return "Early Month"
        elif day <= 20:
            return "Mid Month"
        else:
            return "End of Month"

    data["month_phase"] = data[date_column].dt.day.apply(categorize_month_phase)
    return data


def mark_promotions(data):
    """
    Mark promotion days.
    :param data: pandas DataFrame
    :return: DataFrame with promotion days marked
    """
    # Define promotion dates
    promo_dates = {(6, 18), (11, 11), (12, 12)}  # 618  # Singles Day  # Double Twelve
    # Mark promotion days
    data["is_promo"] = data["order_date"].apply(
        lambda x: (x.month, x.day) in promo_dates
    )
    return data


def perform_feature_engineering(data, date_column):
    """
    Perform feature engineering.
    :param data: pandas DataFrame
    :param date_column: datetime column name
    :return: DataFrame with engineered features
    """
    data = extract_date_features(data)
    data = add_season_feature(data)
    # data = encode_categorical_features(data)
    # Encode all categorical features
    data = encode_all_categorical_features(data, date_column)
    # Create holiday feature
    unique_years = data[date_column].dt.year.unique()
    data = add_holiday_feature(data, list(unique_years))
    # Create month phase feature
    data = add_month_phase_feature(data, date_column)
    # Mark promotion days
    data = mark_promotions(data)
    return data


def main():
    # Example file path, modify according to your data
    file_path = "../data/processed/processed_data.csv"
    date_column = "order_date"  # Change to your actual date column name

    # Load data
    data = pd.read_csv(file_path)

    # Convert date column
    data[date_column] = pd.to_datetime(data[date_column])

    # Perform feature engineering
    feature_engineered_data = perform_feature_engineering(data, date_column)

    # Optionally, save or return the feature engineered data
    feature_engineered_data.to_csv(
        "../data/processed/feature_engineered_data.csv", index=False
    )
    return feature_engineered_data


if __name__ == "__main__":
    main()
