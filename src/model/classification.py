from typing import NamedTuple, Union, Type
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


class ClasModel(NamedTuple):
    mdl: Union[
        Type[LogisticRegression],
        Type[DecisionTreeClassifier],
        Type[RandomForestClassifier],
        Type[GradientBoostingClassifier],
        Type[AdaBoostClassifier],
        Type[SVC],
        Type[KNeighborsClassifier],
        Type[MLPClassifier],
    ]
    fmt: str


CLASSIFIER = {
    "logistic": ClasModel(mdl=LogisticRegression, fmt="sklearn"),
    "decision_tree": ClasModel(mdl=DecisionTreeClassifier, fmt="sklearn"),
    "svc": ClasModel(mdl=SVC, fmt="sklearn"),
    "random_forest": ClasModel(mdl=RandomForestClassifier, fmt="sklearn"),
    "gradient_boosting": ClasModel(mdl=GradientBoostingClassifier, fmt="sklearn"),
    "ada_boost": ClasModel(mdl=AdaBoostClassifier, fmt="sklearn"),
    "knn": ClasModel(mdl=KNeighborsClassifier, fmt="sklearn"),
    "mlp": ClasModel(mdl=MLPClassifier, fmt="sklearn"),
}


