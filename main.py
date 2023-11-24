from src.data_preprocessing import load_data, preprocess_data
from src.eda import perform_eda
from src.feature_engineering import perform_feature_engineering
from src.models import train_and_evaluate, build_pipeline, optimize_model

# from src.utils import perform_statistical_analysis

from sklearn.model_selection import train_test_split


def main():
    # Example file path
    train_data_path = "data/raw/order_train0.csv"

    # 数据加载和清洗
    train_data = load_data(train_data_path)
    if train_data is not None:
        train_data = preprocess_data(train_data)

    # if train_data is not None:
    #     perform_eda(train_data)

    # 特征工程
    if train_data is not None:
        train_data = perform_feature_engineering(train_data, "order_date")
        print("特征工程后的数据预览：")
        print(train_data.head())

    # # 统计分析
    # if train_data is not None:
    #     perform_statistical_analysis(train_data)

    # Split data into features and target
    x = train_data[
        [
            "year",
            "month",
            "sales_region_code",
            "first_cate_code",
            "sales_chan_name",
            "item_price",
        ]
    ]
    y = train_data["ord_qty"]

    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # Build and train the model
    pipeline = build_pipeline()

    # Define parameters for GridSearchCV
    param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
    }
    # Optimize the model
    optimized_model = optimize_model(pipeline, param_grid, x_train, y_train)

    evaluation_results = train_and_evaluate(pipeline, x_train, x_test, y_train, y_test)

    print(f"Mean Squared Error: {evaluation_results['MSE']}")
    print(f"Mean Absolute Error: {evaluation_results['MAE']}")
    print(f"R2 Score: {evaluation_results['R2']}")

    evaluation_results_op = train_and_evaluate(
        optimized_model, x_train, x_test, y_train, y_test
    )
    print(f"Mean Squared Error (optimized): {evaluation_results_op['MSE']}")
    print(f"Mean Absolute Error (optimized): {evaluation_results_op['MAE']}")
    print(f"R2 Score (optimized): {evaluation_results_op['R2']}")


if __name__ == "__main__":
    main()
