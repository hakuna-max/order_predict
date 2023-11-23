# 产品订单的数据分析与需求预测

## Project file structure
```shell
order_predict/
│
├── data/
│   ├── raw/                   # 原始数据，例如 order_train0.csv 和 predict_sku0.csv
│   └── processed/             # 存放数据清洗和预处理后的数据
│
├── notebooks/                 # Jupyter notebook，用于执行和展示探索性数据分析和其他分析报告
│
├── src/                       # 源代码
│   ├── __init__.py
│   ├── data_preprocessing.py  # 数据清洗和预处理的脚本
│   ├── eda.py                 # 探索性数据分析的脚本
│   ├── feature_engineering.py # 特征工程的脚本
│   ├── models.py              # 模型构建和训练
│   └── utils.py               # 工具函数，如绘图、统计分析等
│
├── tests/                     # 测试脚本
│   ├── __init__.py
│   └── test_data_preprocessing.py
│
├── pyproject.toml          # 项目依赖
│
├── .gitignore                # Git 忽略文件配置
│
└── README.md                 # 项目说明文件

```
