from typing import NamedTuple, Union, Type
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
mean_squared_error,
mean_absolute_error,
root_mean_squared_error,
r2_score,
mean_absolute_percentage_error,
)


class RegModel(NamedTuple):
    mdl: Union[
        Type[LinearRegression],
        Type[Ridge],
        Type[Lasso],
        Type[MLPRegressor],
        Type[SVR],
        Type[DecisionTreeRegressor],
        Type[RandomForestRegressor],
        Type[GradientBoostingRegressor],
        Type[AdaBoostRegressor],
        Type[KNeighborsRegressor],
    ]
    fmt: str


REGRESSOR = {
    "linear": RegModel(mdl=LinearRegression, fmt="sklearn"),
    "ridge": RegModel(mdl=Ridge, fmt="sklearn"),
    "lasso": RegModel(mdl=Lasso, fmt="sklearn"),
    "mlp": RegModel(mdl=MLPRegressor, fmt="sklearn"),
    "svr": RegModel(mdl=SVR, fmt="sklearn"),
    "decision_tree": RegModel(mdl=DecisionTreeRegressor, fmt="sklearn"),
    "random_forest": RegModel(mdl=RandomForestRegressor, fmt="sklearn"),
    "gradient_boosting": RegModel(mdl=GradientBoostingRegressor, fmt="sklearn"),
    "ada_boost": RegModel(mdl=AdaBoostRegressor, fmt="sklearn"),
    "knn": RegModel(mdl=KNeighborsRegressor, fmt="sklearn"),
}


REGRESSION_SCORE = {
    "mse": mean_squared_error,
    "rmse": root_mean_squared_error,
    "mae": mean_absolute_error,
    "r2": r2_score,
    "mape": mean_absolute_percentage_error,
}


