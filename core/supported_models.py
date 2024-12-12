from core.models import (
    RandomForestClass,
    KerasNN,
    RandomForestReg,
    LinearReg,
    StochasticGradDescentClass,
    ClusterKmeans,
    BG_NBD,
    GG_NBD,
    XGboostClassifier,
    XGboostRegression,
    DotEmbedding,
)


SUPPORTED_MODEL_TYPES = {
    "random_forest_regression": RandomForestReg,
    "random_forest_classification": RandomForestClass,
    "neural_network_classification": KerasNN,
    "linear_regression": LinearReg,
    "logistic_regression": StochasticGradDescentClass,
    "kmeans_clustering": ClusterKmeans,
    "beta_geometric": BG_NBD,
    "gamma_gamma": GG_NBD,
    "xgboost_classification": XGboostClassifier,
    "xgboost_regression": XGboostRegression,
    "dot_embedding": DotEmbedding,
}
