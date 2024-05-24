import re

import pandas as pd
import os.path
import fnmatch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn import model_selection as cross_validation
from sklearn.metrics import make_scorer, accuracy_score, check_scoring
import sys
import os

# %pip install fairlearn
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity

RANDOM_SEEDS = [87656123, 741246123, 292461935, 502217591, 9327935, 2147631, 2010588, 5171154, 6624906, 5136170]


def get_data():
    X_train = pd.read_csv('adult/preprocessed/x_train.csv')
    s_train = pd.read_csv('adult/preprocessed/s_train.csv')
    y_train = pd.read_csv('adult/preprocessed/y_train.csv')

    X_valid = pd.read_csv('adult/preprocessed/x_valid.csv')
    s_valid = pd.read_csv('adult/preprocessed/s_valid.csv')
    y_valid = pd.read_csv('adult/preprocessed/y_valid.csv')

    X_test = pd.read_csv('adult/preprocessed/x_test.csv')
    s_test = pd.read_csv('adult/preprocessed/s_test.csv')
    y_test = pd.read_csv('adult/preprocessed/y_test.csv')

    return (X_train, s_train, y_train), (X_valid, s_valid, y_valid), (X_test, s_test, y_test)

def get_decoded_data(k, fairness, scale, alpha):
    decoded_train = pd.read_csv(f'results/adult/{k}_components/{fairness}/hsic/{scale}/{alpha}/projected_data.csv')
    decoded_test = pd.read_csv(f'results/adult_test/{k}_components/{fairness}/hsic/{scale}/{alpha}/projected_data.csv')

    return decoded_train, decoded_test

def get_weights(s, y):
    unique_S = np.unique(s)
    unique_C = np.unique(y)
    print("unique_S: ", unique_S)
    print("unique_C: ", unique_C)
    W = dict()
    for x in unique_S:
        for c in unique_C:
            if x not in W:
                W[x] = dict()
            s_in_D = np.count_nonzero(s == x)
            c_in_D = np.count_nonzero(y == c)
            # print("s == x: ", s == x)
            # print("y == c: ", y == c)
            # print("np.logical_and(s == x, y == c): ", np.logical_and(np.array(s == x).flatten(), np.array(y == c).flatten()))
            s_and_c_in_D = np.count_nonzero(np.logical_and(s == x, y == c))
            W[x][c] = s_in_D * c_in_D / (s.shape[0] * s_and_c_in_D)
    print("s.shape[0]: ", s.shape[0])
    print("W[s[0]][y[0]]: ", W[s[0]][y[0]])
    weights = [W[s[i]][y[i]] for i in range(s.shape[0])]
    return weights

def compute_accuracy_pvalue(Y, predictions, Xcontrol):
    correct = np.sum(Y == predictions)
    acc = correct * 1. / Y.shape[0]
    acc_sensitive = np.zeros(np.unique(Xcontrol).shape[0])
    ii = 0
    for v in np.unique(Xcontrol):
        idx_ = Xcontrol == v
        acc_sensitive[ii] = np.sum(Y[idx_] == predictions[idx_,]) / (np.sum(idx_) * 1.)
        ii = ii + 1
    return acc, acc_sensitive  # , pvalue

def compute_fpr_fnr(Y, predictions, Xcontrol):
    fp = np.sum(np.logical_and(Y == 0.0, predictions == +1.0))  # something which is -ve but is misclassified as +ve
    fn = np.sum(np.logical_and(Y == +1.0, predictions == 0.0))  # something which is +ve but is misclassified as -ve
    tp = np.sum(
        np.logical_and(Y == +1.0, predictions == +1.0))  # something which is +ve AND is correctly classified as +ve
    tn = np.sum(
        np.logical_and(Y == 0.0, predictions == 0.0))  # something which is -ve AND is correctly classified as -ve
    fpr_all = float(fp) / float(fp + tn)
    fnr_all = float(fn) / float(fn + tp)
    tpr_all = float(tp) / float(tp + fn)
    tnr_all = float(tn) / float(tn + fp)

    fpr_fnr_tpr_sensitive = np.zeros((4, np.unique(Xcontrol).shape[0]))
    ii = 0
    for v in np.unique(Xcontrol):
        idx_ = Xcontrol == v
        fp = np.sum(np.logical_and(Y[idx_] == 0.0,
                                   predictions[idx_] == +1.0))  # something which is -ve but is misclassified as +ve
        fn = np.sum(np.logical_and(Y[idx_] == +1.0,
                                   predictions[idx_] == 0.0))  # something which is +ve but is misclassified as -ve
        tp = np.sum(np.logical_and(Y[idx_] == +1.0, predictions[
            idx_] == +1.0))  # something which is +ve AND is correctly classified as +ve
        tn = np.sum(np.logical_and(Y[idx_] == 0.0, predictions[
            idx_] == 0.0))  # something which is -ve AND is correctly classified as -ve
        fpr = float(fp) / float(fp + tn)
        fnr = float(fn) / float(fn + tp)
        tpr = float(tp) / float(tp + fn)
        tnr = float(tn) / float(tn + fp)
        fpr_fnr_tpr_sensitive[0, ii] = fpr
        fpr_fnr_tpr_sensitive[1, ii] = fnr
        fpr_fnr_tpr_sensitive[2, ii] = tpr
        fpr_fnr_tpr_sensitive[3, ii] = tnr
        ii = ii + 1
    return fpr_all, fnr_all, fpr_fnr_tpr_sensitive

def get_classifier(classifier, reg):
    return classifier(C=reg, dual=False, tol=1.0e-6, random_state=888)

def test(data_train, data_valid, data_test, decoded_train, decoded_test, classifier, fairness=None):

    X_train, s_train, y_train = data_train
    X_valid, s_valid, y_valid = data_valid
    X_test, s_test, y_test = data_test
    tpr_diff = []
    fpr_dif = []
    acc_ = []

    # perform classification here with X and Xtilde
    reg_array = [10 ** i for i in range(7)]
    n_splits = 3
    cv = cross_validation.StratifiedKFold(n_splits=n_splits, random_state=888, shuffle=True)
    # with Xtilde
    print("with Xtilde for all iterations")
    cv_scores = np.zeros((len(reg_array), n_splits))
    for i, reg_const in enumerate(reg_array):
        if fairness == 'fl':
            clf1 = get_classifier(classifier, reg_const)
            tprp = TruePositiveRateParity(difference_bound=0.01)
            # tprp.load_data(decoded_train, y_train, sensitive_features=s_train)
            clf = ExponentiatedGradient(clf1, tprp)
            cv_scores[i] = cross_validation.cross_val_score(
                clf, decoded_train, np.array(y_train).flatten(),
                cv=cv, scoring=make_scorer(accuracy_score), fit_params={'sensitive_features': s_train})
        else:
            clf = get_classifier(classifier, reg_const)
            cv_scores[i] = cross_validation.cross_val_score(
                clf, decoded_train, np.array(y_train).flatten(),
                cv=cv)
    cv_mean = np.mean(cv_scores, axis=1)
    reg_best = reg_array[np.argmax(cv_mean)]
    print("Regularization ", reg_best)
    if fairness == 'kc':
        clf = get_classifier(classifier, reg_best)
        weights = get_weights(np.array(s_train).flatten(), 
                              np.array(y_train).flatten())
        clf.fit(decoded_train, np.array(y_train).flatten(), sample_weight=weights)
    elif fairness == 'fl':
        clf1 = get_classifier(classifier, reg_best)
        tprp = TruePositiveRateParity(difference_bound=0.01)
        clf = ExponentiatedGradient(clf1, tprp)
        clf.fit(decoded_train, np.array(y_train).flatten(), sensitive_features=np.array(s_train).flatten())
    else:
        clf = get_classifier(classifier, reg_best)
        clf.fit(decoded_train, np.array(y_train).flatten())

    predictions = clf.predict(decoded_test)
    # performance measurement
    acc, acc_sensitive = compute_accuracy_pvalue(np.array(y_test).flatten(), predictions,
                                                np.array(s_test).flatten())
    print(f'{classifier} Accuracy: %.2f%%' % (acc * 100.))
    print("per sensitive value: %.2f, %.2f, (%.2f)" % (
    acc_sensitive[0] * 100., acc_sensitive[1] * 100., (acc_sensitive[0] - acc_sensitive[1]) * 100.))
    fpr, fnr, fpr_fnr_tpr_sensitive = compute_fpr_fnr(np.array(y_test).flatten(), predictions,
                                                        np.array(s_test).flatten())
    print(f'{classifier} FPR and FNR: %.2f, %.2f' % (fpr * 100., fnr * 100.))
    print("TPR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[2, 0] * 100., fpr_fnr_tpr_sensitive[2, 1] * 100.,
    (fpr_fnr_tpr_sensitive[2, 0] - fpr_fnr_tpr_sensitive[2, 1]) * 100.))
    print("FPR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[0, 0] * 100., fpr_fnr_tpr_sensitive[0, 1] * 100.,
    (fpr_fnr_tpr_sensitive[0, 0] - fpr_fnr_tpr_sensitive[0, 1]) * 100.))
    print("FNR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[1, 0] * 100., fpr_fnr_tpr_sensitive[1, 1] * 100.,
    (fpr_fnr_tpr_sensitive[1, 0] - fpr_fnr_tpr_sensitive[1, 1]) * 100.))
    print("TNR per sensitive value: %.2f, %.2f, (%.2f)" % (
    fpr_fnr_tpr_sensitive[3, 0] * 100., fpr_fnr_tpr_sensitive[3, 1] * 100.,
    (fpr_fnr_tpr_sensitive[3, 0] - fpr_fnr_tpr_sensitive[3, 1]) * 100.))
    print("\n")
    acc_.append(acc)
    tpr_diff.append((fpr_fnr_tpr_sensitive[2, 0] - fpr_fnr_tpr_sensitive[2, 1]) * 100.)
    fpr_dif.append((fpr_fnr_tpr_sensitive[0, 0] - fpr_fnr_tpr_sensitive[0, 1]) * 100.)

    x_column_names = ["age",
                        "education-num",
                        "capital-gain",
                        "capital-loss",
                        "hours-per-week",
                        "workclass_Federal-gov",
                        "workclass_Local-gov",
                        "workclass_Never-worked",
                        "workclass_Private",
                        "workclass_Self-emp-inc",
                        "workclass_Self-emp-not-inc",
                        "workclass_State-gov",
                        "workclass_Without-pay",
                        "education_10th",
                          "education_11th",
                          "education_12th",
                          "education_1st-4th",
                          "education_5th-6th",
                          "education_7th-8th",
                          "education_9th",
                          "education_Assoc-acdm",
                          "education_Assoc-voc",
                          "education_Bachelors",
                          "education_Doctorate",
                          "education_HS-grad",
                          "education_Masters",
                          "education_Preschool",
                          "education_Prof-school",
                          "education_Some-college",
                          "marital-status_Divorced",
                          "marital-status_Married-AF-spouse",
                          "marital-status_Married-civ-spouse",
                          "marital-status_Married-spouse-absent",
                          "marital-status_Never-married",
                          "marital-status_Separated",
                          "marital-status_Widowed",
                          "occupation_Adm-clerical",
                          "occupation_Armed-Forces",
                          "occupation_Craft-repair",
                          "occupation_Exec-managerial",
                          "occupation_Farming-fishing",
                          "occupation_Handlers-cleaners",
                          "occupation_Machine-op-inspct",
                          "occupation_Other-service",
                          "occupation_Priv-house-serv",
                          "occupation_Prof-specialty",
                          "occupation_Protective-serv",
                          "occupation_Sales",
                          "occupation_Tech-support",
                          "occupation_Transport-moving",
                          "relationship_Husband",
                          "relationship_Not-in-family",
                          "relationship_Other-relative",
                          "relationship_Own-child",
                          "relationship_Unmarried",
                          "relationship_Wife",
                          # "sex_Female",
                          # "sex_Male",
                          "race_Amer-Indian-Eskimo",
                          "race_Asian-Pac-Islander",
                          "race_Black",
                          "race_Other",
                          "race_White",
                          "native-country_Cambodia",
                          "native-country_Canada",
                          "native-country_China",
                          "native-country_Columbia",
                          "native-country_Cuba",
                          "native-country_Dominican-Republic",
                          "native-country_Ecuador",
                          "native-country_El-Salvador",
                          "native-country_England",
                          "native-country_France",
                          "native-country_Germany",
                          "native-country_Greece",
                          "native-country_Guatemala",
                          "native-country_Haiti",
                          "native-country_Holand-Netherlands",
                          "native-country_Honduras",
                          "native-country_Hong",
                          "native-country_Hungary",
                          "native-country_India",
                          "native-country_Iran",
                          "native-country_Ireland",
                          "native-country_Italy",
                          "native-country_Jamaica",
                          "native-country_Japan",
                          "native-country_Laos",
                          "native-country_Mexico",
                          "native-country_Nicaragua",
                          "native-country_Outlying-US(Guam-USVI-etc)",
                          "native-country_Peru",
                          "native-country_Philippines",
                          "native-country_Poland",
                          "native-country_Portugal",
                          "native-country_Puerto-Rico",
                          "native-country_Scotland",
                          "native-country_South",
                          "native-country_Taiwan",
                          "native-country_Thailand",
                          "native-country_Trinadad&Tobago",
                          "native-country_United-States",
                          "native-country_Vietnam",
                          "native-country_Yugoslavia",
                          ]

    s_column_name = ["sex_Male"]
    y_column_name = ["salary"]

    print(np.array(acc_))
    print(np.array(tpr_diff))
    print(np.array(fpr_dif))

    return True