import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import holidays


def extract_date_features(data):
    """
    从日期中提取特征
    :param data: pandas DataFrame
    :return: 增加日期特征的DataFrame
    """
    data['year'] = data['order_date'].dt.year
    data['month'] = data['order_date'].dt.month
    data['day'] = data['order_date'].dt.day
    data['weekday'] = data['order_date'].dt.weekday
    # 根据季节性特征（北半球）分配季节
    data['season'] = data['month'].apply(lambda x: (x % 12 + 3) // 3)
    return data


def encode_categorical_features(data):
    """
    对分类特征进行编码
    :param data: pandas DataFrame
    :return: 编码后的DataFrame
    """
    # 独热编码示例 - 对 'sales_chan_name' 进行编码
    encoder = OneHotEncoder(sparse_output=False)
    sales_chan_encoded = encoder.fit_transform(data[['sales_chan_name']])
    sales_chan_encoded_df = pd.DataFrame(sales_chan_encoded, columns=encoder.get_feature_names_out(['sales_chan_name']))
    data = pd.concat([data, sales_chan_encoded_df], axis=1).drop('sales_chan_name', axis=1)
    return data


# def encode_all_categorical_features(data):
#     """
#     对所有分类特征进行编码
#     :param data: pandas DataFrame
#     :return: 编码后的DataFrame
#     """
#     categorical_columns = data.select_dtypes(include=['object']).columns
#     for col in categorical_columns:
#         if col != date:  # 假设 date 是日期列的名称
#             encoder = OneHotEncoder(sparse_output=False)
#             encoded = encoder.fit_transform(data[[col]])
#             encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]))
#             data = pd.concat([data, encoded_df], axis=1).drop(col, axis=1)
#     return data

def add_season_feature(data):
    """
    添加季节特征，假设使用北半球的季节划分：春季（3-5月），夏季（6-8月），秋季（9-11月），冬季（12-2月）
    :param data: pandas DataFrame
    :return: 增加季节特征的DataFrame
    """
    def get_season(month):
        if month in [3, 4, 5]:
            return '春季'
        elif month in [6, 7, 8]:
            return '夏季'
        elif month in [9, 10, 11]:
            return '秋季'
        else:
            return '冬季'

    data['season'] = data['month'].apply(get_season)
    return data


def add_holiday_feature(data, years):
    """
    为数据集添加中国节假日标记
    :param data: pandas DataFrame
    :param years: 包含年份的列表
    :return: 带有节假日标记的DataFrame
    """
    cn_holidays = holidays.CN(years=years)
    data['is_holiday'] = data['order_date'].apply(lambda x: x in cn_holidays)
    return data


def add_month_phase_feature(data, date_column):
    """
    为数据添加月初、月中、月末标记
    :param data: pandas DataFrame
    :param date_column: 用于提取月份阶段的日期列名称
    :return: DataFrame
    """

    def categorize_month_phase(day):
        if day <= 10:
            return '月初'
        elif day <= 20:
            return '月中'
        else:
            return '月末'

    data['month_phase'] = data[date_column].dt.day.apply(categorize_month_phase)
    return data


def mark_promotions(data):
    """
    标记促销活动日
    :param data: pandas DataFrame
    :return: 带有促销活动标记的DataFrame
    """
    # 定义促销活动日期
    promo_dates = {
        (6, 18),  # 618
        (11, 11),  # 双十一
        (12, 12)   # 双十二
    }

    # 标记促销活动日
    data['is_promo'] = data['order_date'].apply(lambda x: (x.month, x.day) in promo_dates)
    return data


def perform_feature_engineering(data, date_column):
    """
    执行特征工程的总函数
    :param data: pandas DataFrame
    :param date: datetime
    :return: 经过特征工程处理的DataFrame
    """
    data = extract_date_features(data)
    data = add_season_feature(data)
    data = encode_categorical_features(data)

    # 创建节假日特征
    unique_years = data[date_column].dt.year.unique()
    unique_years = list(unique_years)
    data = add_holiday_feature(data, unique_years)

    # 创建月份阶段特征
    data = add_month_phase_feature(data, date_column)

    # 标记促销活动日
    data = mark_promotions(data)

    return data
