from scipy.stats import distributions
import preprocessing
import numpy as np
import sklearn

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import sklearn.tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import uniform






def single_tree_modelling_experiment(X_train, y_train, X_test, y_test, grid_search=None, random_search=None):
    best_parameters = {'ccp_alpha': 0.005}
    tree = DecisionTreeClassifier()
    cv = KFold(n_splits=4)
    results = {}

    if grid_search is not None:
        gs = GridSearchCV(estimator=tree, param_grid=grid_search, cv=cv)
        gs.fit(X_train, y_train)
        best_parameters = {**best_parameters, **gs.best_params_}
        results['gs_cv'] = gs.cv_results_

    if random_search is not None:
        rs = RandomizedSearchCV(estimator=tree, param_distributions=random_search, cv=cv, n_iter=10)
        rs.fit(X_train, y_train)
        # print('random search params:', rs.best_params_)
        best_parameters = {**best_parameters, **rs.best_params_}
        results['rs_cv'] = rs.cv_results_

    best_tree = DecisionTreeClassifier(**best_parameters)
    results['tree'] = best_tree
    best_tree.fit(X_train, y_train)
    y_hat_test = best_tree.predict(X_test)
    y_hat_train = best_tree.predict(X_train)

    results['confusion_train'] = confusion_matrix(y_hat_train, y_train)
    results['confusion_test'] = confusion_matrix(y_hat_test, y_test)
    # print(best_parameters)
    # print("Test confusion:")
    # print(confusion_matrix(y_hat_test, y_test))
    # print("Train confusion:")
    # print(confusion_matrix(y_hat_train, y_train))
    return results

def single_tree_alpha_path(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    plt.show()

def random_forest_experiment(X_train, y_train, X_test, y_test, grid_search=None, random_search=None, given_parameters=None):
    best_parameters = given_parameters
    rf = RandomForestClassifier()
    cv = KFold(n_splits=4)
    results = {}

    if grid_search is not None:
        gs = GridSearchCV(estimator=rf, param_grid=grid_search, cv=cv)
        gs.fit(X_train, y_train)
        best_parameters = {**best_parameters, **gs.best_params_}
        results['gs_cv'] = gs.cv_results_

    if random_search is not None:
        rs = RandomizedSearchCV(estimator=rf, param_distributions=random_search, cv=cv, n_iter=10)
        rs.fit(X_train, y_train)
        # print('random search params:', rs.best_params_)
        best_parameters = {**best_parameters, **rs.best_params_}
        results['rs_cv'] = rs.cv_results_

    best_rf = RandomForestClassifier(**best_parameters)
    results['rf'] = best_rf
    best_rf.fit(X_train, y_train)
    y_hat_test = best_rf.predict(X_test)
    y_hat_train = best_rf.predict(X_train)

    results['confusion_train'] = confusion_matrix(y_hat_train, y_train)
    results['confusion_test'] = confusion_matrix(y_hat_test, y_test)
    results['best_parameters'] = best_parameters
    results['accuracy_train'] = accuracy_score(y_hat_train, y_train)
    results['accuracy_test'] = accuracy_score(y_hat_test, y_test)
    return results

# df_read = preprocessing.read_into_pandas_dataframe()
# df_corpus, corpus_tm, X_train_corpus, y_train_corpus, X_test_corpus, y_test_corpus, X_dev_folds, y_dev_folds = preprocessing.preprocessing(df_read,
#                 del_punkt=True,
#                 lower_case=True,
#                 del_numbers=True,
#                 del_stopwords=True,
#                 stemming=True,
#                 pos_tagging=False,
#                 ngrams=2)

# parameters = {'max_depth': [2, 10], 'min_samples_split':[2, 10], 'min_samples_leaf':[2, 10], 'ccp_alpha': [0, 100.0]}
# grid_search = {'ccp_alpha': [0.006]}
# random_search = None
#

if __name__ == '__main__':
    grid_search = None
    random_search = {'ccp_alpha': uniform(loc=0.0, scale=0.009)}
    results = single_tree_modelling_experiment(X_train_corpus,
                                    y_train_corpus,
                                    X_test_corpus,
                                    y_test_corpus,
                                    grid_search=grid_search,
                                    random_search=random_search)

    # single_tree_alpha_path(X_train_corpus,
    #                                 y_train_corpus,
    #                                 X_test_corpus,
    #                                 y_test_corpus)
