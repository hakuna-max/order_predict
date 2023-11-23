import pandas as pd


def perform_statistical_analysis(data):
    """
    对DataFrame执行基本的统计分析
    :param data: pandas DataFrame
    :return: None
    """
    print("基本统计信息：")
    print(data.describe())

    # 这里您可以添加更多的统计分析，如分位数、众数等
    # 例如：print(data.quantile([0.25, 0.5, 0.75]))
    # 或者：print(data.mode())
