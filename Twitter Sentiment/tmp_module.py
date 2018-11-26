import pandas as pd
import numpy as np
from scipy.special import expit
from collections import Counter
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import validation_curve as skl_validation_curve
from sklearn.model_selection import learning_curve as skl_learning_curve
from sklearn.utils import shuffle
from matplotlib import pyplot

import os
print(os.path.dirname(os.path.realpath(__file__)))


class PoissonNaiveBayes(BaseEstimator, RegressorMixin):
    def __init__(self, n=1):
        self.n = n
        
    def fit(self, X, y):
        self.n0_ = (y == 0).sum() # S0
        self.n1_ = (y == 1).sum() # S1
        w0 = pd.Series(X[y == 0].sum()) # {tweet|tweet\in S_0, x_i \in tweet}
        w1 = pd.Series(X[y == 1].sum()) # {tweet|tweet\in S_0, x_i \in tweet}
        self.theta0_ = w0.mean() / self.n0_ # theta0
        self.theta1_ = w1.mean() / self.n1_ # theta1
        self.l0_ = (w0 + self.n * self.theta0_) / (self.n0_ + self.n) # \lambda_i0
        self.l1_ = (w1 + self.n * self.theta0_) / (self.n1_ + self.n) # \lambda_i1
        return self
    
    def predict_proba(self, X):
        # we assume each X[i] is a counter
        L0 = self.l0_.sum()
        L1 = self.l1_.sum()
        p0 = np.log(self.n0_) - L0 + pd.Series(X).map(lambda x: np.log((self.l0_.reindex(x.keys())) * pd.Series(x)).fillna(self.theta0_).sum())
        p1 = np.log(self.n1_) - L1 + pd.Series(X).map(lambda x: np.log((self.l1_.reindex(x.keys())) * pd.Series(x)).fillna(self.theta1_).sum())
        return expit(p0 - p1)
    
    def predict(self, X):
        return self.predict_proba(X) > .5


def validation_curve(estimator, X, y, scoring, param_name,
                     n_min, n_max, n_points=10, log_scale=False,
                     cv=None, n_jobs=4, train_curve=True):
 
    args = (n_min, n_max, n_points)
    param_range = np.logspace(*args) if log_scale else np.linspace(*args)

    X, y = shuffle(X, y)
    pyplot.title("Validation Curve (%s)" % param_name)
    pyplot.xlabel("Parameter")
    pyplot.ylabel("Score")

    train_scores, test_scores = skl_validation_curve(
        estimator, X, y,
        param_name=param_name,
        param_range=param_range,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs, verbose=10)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    if log_scale:
        pyplot.semilogx(
            param_range, test_scores_mean,
            'o-', color="g", label='test scores')
        if train_curve:
            pyplot.semilogx(
                param_range, train_scores_mean,
                'o-', color="r", label='train scores')
    else:
        pyplot.plot(
            param_range, test_scores_mean,
            'o-', color="g", label='test scores')
        if train_curve:
            pyplot.plot(
                param_range, train_scores_mean,
                'o-', color="r", label='train scores')

    pyplot.legend(loc='upper right')


def learning_curve(estimator, X, y, scoring='roc_auc',
                   n_min=.1, n_max=1, n_points=10, log_scale=False,
                   cv=None, n_jobs=4):

    args = (n_min, n_max, n_points)
    train_sizes = np.logspace(*args) if log_scale else np.linspace(*args)

    X, y = shuffle(X, y)
    pyplot.figure()
    pyplot.title("Leaning Curve")
    pyplot.xlabel("Training samples")
    pyplot.ylabel("Score")
    train_sizes, train_scores, test_scores = skl_learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    pyplot.grid()

    pyplot.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1, color="r")
    pyplot.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1, color="g")
    if log_scale:
        pyplot.semilogx(
            train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
        pyplot.semilogx(
            train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")
    else:
        pyplot.plot(
            train_sizes, train_scores_mean, 'o-', color="r",
            label="Training score")
        pyplot.plot(
            train_sizes, test_scores_mean, 'o-', color="g",
            label="Cross-validation score")

    pyplot.xlim([0, X.shape[0]])

    pyplot.legend(loc="best")