import pandas as pd


def load_data(file_path):
    """
    加载数据集
    :param file_path: 数据集文件的路径
    :return: 返回加载的DataFrame
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到。")
        return None
    except Exception as e:
        print(f"加载数据时发生错误：{e}")
        return None


def main():
    # 修改为您的数据文件路径
    train_data_path = "../data/raw/order_train0.csv"
    predict_data_path = "../data/raw/predict_sku0.csv"

    # 加载数据
    train_data = load_data(train_data_path)
    predict_data = load_data(predict_data_path)

    # 基本的数据预览
    if train_data is not None:
        print("训练数据预览：")
        print(train_data.head())

    if predict_data is not None:
        print("预测数据预览：")
        print(predict_data.head())


if __name__ == "__main__":
    main()
