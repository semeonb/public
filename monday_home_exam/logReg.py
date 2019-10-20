import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn import metrics


def modelBuild(X, Y, penalty, max_iter, tol, seed, lossFunction='log', n_jobs=2, alpha=0.00001):
    log_model = SGDClassifier(loss=lossFunction, penalty=penalty, n_jobs=n_jobs,
                              warm_start=True, shuffle=True, max_iter=max_iter,
                              tol=tol, alpha=alpha, random_state=seed)
    log_model.fit(X, Y)

    return log_model


def calcLogLoss(X, y, model):
    pred = model.predict_proba(X)
    return metrics.log_loss(y, pred)