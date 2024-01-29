import numpy as np
import pandas as pd
import os
import argparse
import pickle

from copy import copy
from bisect import bisect_left
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from scipy.stats import gamma
from scipy.special import softmax

# You might have to pip install fairlearn
# pip install fairlearn
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity


def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    
    From https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value/12141511#12141511
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before
    

def scrub_data(df, pc, target, pc_range=[0, 1], target_range=[0, 1], inplace=False):
    if inplace:
        target_data = df
    else:
        target_data = copy(df)
    target_data[pc] = target_data[pc].map(lambda x : take_closest(pc_range, x))
    target_data[target] = target_data[target].map(lambda x : take_closest(target_range, x))
    return target_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scale',
                        help='the directory in which to load the projected data, eg "scale_0.1"',
                        type=str)
    parser.add_argument('-f', '--fold',
                        help='which fold of the data to use (can be between 0 and 2 inclusive)',
                        type=int, choices=[0, 1, 2])
    parser.add_argument('-i', '--regparam',
                        help='the regularisation strength to use for this run (10^i, choose i from 0-6)',
                        type=int, choices=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument('-s', '--save',
                        help='whether to save the result (default is True)',
                        type=bool, default=True)
    args = parser.parse_args()
    datapath = 'results/adult/'
    random_seed = 0
    original_data = pd.read_csv('adult/adult_trimmed.data')
    pc = original_data['sex'] # protected characteristic
    y = original_data['income'] # target feature
    filepath = os.path.join('results', 'adult_full', '2_components', 'independence', 'hsic')
    fairness_penalties = []
    accuracies = []
    dataset = pd.read_csv(os.path.join(filepath, args.scale, 'projected_data.csv'))
    dataset = scrub_data(dataset, 'Column3', 'Column4')
    X = dataset.loc[:, dataset.columns != 'Column4']
    kfold = KFold(n_splits=3, shuffle=True, random_state=random_seed)
    if args.fold:
        folds = [kfold.split(X, y=y)[args.fold]]
    else:
        folds = kfold.split(X, y=y)
    if args.regparam:
        regparams = [args.regparam]
    else:
        regparams = range(7)
    num_fold = args.fold if args.fold else 0
    for train, test in folds:
        X_train = X.loc[train, :]
        y_train = y[train]
        X_test = X.loc[test, :]
        y_test = y[test]
        for p in regparams:
            print(f"Parameters:")
            print(f"* k = 2")
            print(f"* fairness = independence")
            print(f"* independence criterion = hsic")
            print(f"* scale = {args.scale}")
            print(f"* fold = {num_fold}")
            print(f"* regularisation parameter = {p}")
            print("--------------------")
            num_fold += 1
            svm = SVC(C=10**(-p))
            tprp = TruePositiveRateParity(difference_bound=0.01)
            egc = ExponentiatedGradient(svm, tprp)
            egc.fit(X_train, y_train, sensitive_features=X_train['Column3'])
            y_pred = egc.predict(X_test)
            accuracy = float(sum(np.where(y_pred == y_test, 1, 0))) / float(len(y_test))
            accuracies.append(accuracy)
            print(f"Accuracy: {accuracy}")
            pos_indices = [i for i, y in enumerate(y_test) if y]
            X_test_copy = copy(X_test)
            X_test_copy['y_pred'] = y_pred
            pos_labels = X_test_copy.iloc[pos_indices, :]
            A_group = pos_labels.loc[pos_labels['Column3'] == 1]
            B_group = pos_labels.loc[pos_labels['Column3'] == 0]
            A_opp_rate = len([x for x in A_group['y_pred'] if x == 1]) / len(A_group['y_pred'])
            B_opp_rate = len([x for x in B_group['y_pred'] if x == 1]) / len(B_group['y_pred'])
            equal_opp = abs(A_opp_rate - B_opp_rate)
            print(f"Equal opportunity: {equal_opp}")
            print()
            fairness_penalties.append(equal_opp)
    if args.save:
        filename = args.scale
        if args.fold:
            filename += f"_fold-{args.fold}"
        if args.regparam:
            filename += f"_regparam-{args.regparam}"
        accuracy_file = filename + '_accuracies'
        fairness_file = filename + '_fairness'
        with open(os.path.join('benchmarks', accuracy_file), 'wb') as afp:
            pickle.dump(accuracies, afp)
        with open(os.path.join('benchmarks', fairness_file), 'wb') as ffp:
            pickle.dump(fairness_penalties, ffp)