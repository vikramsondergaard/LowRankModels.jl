import test_model
import argparse
import pandas as pd
from sklearn import svm
from adult.adult_dataset import load_dataset
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx', type=int, required=False)
    parser.add_argument('-x', '--train_path', type=str, required=False)
    parser.add_argument('-y', '--test_path', type=str, required=False)
    args = parser.parse_args()

    random_state = test_model.RANDOM_SEEDS[args.idx]
    validation_size = 2000
    feature_splits = 'sex_salary'
    i_scaler = MinMaxScaler

    data_train, data_valid, data_test, _ = load_dataset('adult/train.csv',
                                                        'adult/test.csv',
                                                        validation_size,
                                                        random_state,
                                                        'adult',
                                                        feature_splits,
                                                        remake_test=True,
                                                        test_size=15000,
                                                        input_scaler=i_scaler)
    
    if args.train_path is not None: # using a direct path
        decoded_train = pd.read_csv(args.train_path)
    elif args.idx is None:
        raise RuntimeError()
    else: # using an index
        decoded_train = pd.read_csv(f'adult/splits/{args.idx}/results/x_train.csv')

    if args.test_path is not None: # using a direct path
        decoded_test = pd.read_csv(args.train_path)
    elif args.idx is None:
        raise RuntimeError()
    else: # using an index
        decoded_test = pd.read_csv(f'adult/splits/{args.idx}/results/x_test.csv')

    test_model.test(data_train, data_valid, data_test,
        decoded_train, decoded_test, svm.LinearSVC)