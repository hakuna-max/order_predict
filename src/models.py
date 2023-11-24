import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline


def build_pipeline():
    """
    Build a machine learning pipeline with data preprocessing and model.

    Returns:
    sklearn.pipeline.Pipeline: The constructed pipeline.
    """
    # 特征工程已在 feature_engineering.py 中处理，这里可以只构建模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    pipeline = Pipeline([("model", rf_model)])
    return pipeline


def train_and_evaluate(pipeline, x_train, x_test, y_train, y_test):
    """
    Train the model and evaluate its performance.

    Parameters:
    pipeline (sklearn.pipeline.Pipeline): The machine learning pipeline.
    X_train, X_test, y_train, y_test: Training and testing data.

    Returns:
    tuple: Mean squared error and mean absolute error of the model.
    """
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)  # R² score
    return {"MSE": mse, "MAE": mae, "R2": r2}


def optimize_model(pipeline, param_grid, x_train, y_train):
    """
    Optimize the model using GridSearchCV.

    Parameters:
    pipeline (sklearn.pipeline.Pipeline): The machine learning pipeline.
    param_grid (dict): Parameter grid for GridSearchCV.
    x_train, y_train: Training data.

    Returns:
    best_model: The model with optimized parameters.
    """
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring="neg_mean_squared_error",
        verbose=2,
        n_jobs=-1,
    )
    grid_search.fit(x_train, y_train)
    return grid_search.best_estimator_


def main():
    pass


if __name__ == "__main__":
    main()
