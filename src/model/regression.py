from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor


REGRESSOR = {
    "linear": LinearRegression,
    "ridge": Ridge,
    "lasso": Lasso,
    "logistic": LogisticRegression,
    "mlp": MLPRegressor,
    "svr": SVR,
    "decision_tree": DecisionTreeRegressor,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "ada_boost": AdaBoostRegressor,
    "knn": KNeighborsRegressor,
}


