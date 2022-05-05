import numpy as np
from collections import defaultdict
from sklearn.base import clone, BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import validation_curve as skl_validation_curve
from sklearn.model_selection import learning_curve as skl_learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from gensim.models.keyedvectors import KeyedVectors
from sklearn.utils import shuffle
from matplotlib import pyplot


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


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


class TfidfEmbeddingVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, vec):
        self.vec = vec
        self.weight = None

    def fit(self, X, y=None):
        tfidf = TfidfVectorizer()
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])
        return self

    def transform(self, X):
        res = []
        dummy = [np.zeros(self.vec.vector_size)]
        for Xrow in X:
            tmp = []
            for w in Xrow.split(" "):
                if w in self.vec:
                    tmp.append(self.vec[w] * self.weight[w])
            if len(tmp) == 0:
                tmp = dummy
            res.append(np.mean(tmp, axis=0))
        return np.array(res)
