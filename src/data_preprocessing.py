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


def clean_data(data):
    """
    清洗数据集
    :param data: pandas DataFrame
    :return: 清洗后的DataFrame
    """
    # 日期格式处理
    data['order_date'] = pd.to_datetime(data['order_date'], errors='coerce')

    # 价格和数量的合理性检查
    data = data[data['item_price'] > 0]
    data = data[data['ord_qty'] > 0]

    # 缺失值处理（在这个数据集中看起来不是问题）
    # data = data.dropna()

    return data


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

    # 清洗数据
    if train_data is not None:
        train_data = clean_data(train_data)
        print("清洗后的数据预览：")
        print(train_data.head())


if __name__ == "__main__":
    main()
