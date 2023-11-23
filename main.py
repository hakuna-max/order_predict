from src.data_preprocessing import load_data, clean_data
from src.feature_engineering import perform_feature_engineering
from src.utils import perform_statistical_analysis


def main():
    # 示例数据文件路径
    train_data_path = "data/raw/order_train0.csv"

    # 数据加载和清洗
    train_data = load_data(train_data_path)
    if train_data is not None:
        train_data = clean_data(train_data)

    # 特征工程
    if train_data is not None:
        train_data = perform_feature_engineering(train_data)
        print("特征工程后的数据预览：")
        print(train_data.head())

    # 统计分析
    if train_data is not None:
        perform_statistical_analysis(train_data)


if __name__ == "__main__":
    main()
