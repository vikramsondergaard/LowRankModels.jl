import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
from os import makedirs

def _filter_features_by_prefixes(df, prefixes):
    res = []
    for name in df.columns:
        filtered = False
        for pref in prefixes:
            if name.startswith(pref):
                filtered = True
                break
        if not filtered:
            res.append(name)
    return res


def _split_feature_names(df, data_name, feature_split):
    if data_name == "adult":
        if feature_split == "sex_salary":
            x_features = _filter_features_by_prefixes(df, ['sex', 'salary'])
            s_features = ['sex_Male']
            y_features = ['salary_>50K']
        elif feature_split == "race_salary":
            x_features = _filter_features_by_prefixes(
                df, ['race', 'salary'])
            s_features = [
                    'race_Amer-Indian-Eskimo',
                    'race_Asian-Pac-Islander',
                    'race_Black',
                    'race_White',
                ]
            y_features = ['salary_>50K']
        elif feature_split == "sex-race_salary":
            x_features = _filter_features_by_prefixes(
                df, ['sex', 'race', 'salary'])
            s_features = [
                    'sex_Male',
                    'race_Amer-Indian-Eskimo',
                    'race_Asian-Pac-Islander',
                    'race_Black',
                    'race_White',
                ]
            y_features = ['salary_>50K']

        else:
            raise NotImplementedError()
    elif data_name== "nypd":
        if feature_split== "sex_possession":
            x_features = _filter_features_by_prefixes(df, ['sex', 'possession'])
            s_features = ['sex_M']
            y_features = ['possession']
        elif feature_split== "sex-race_possession":
            x_features = _filter_features_by_prefixes(
                df, ['sex', 'race', 'possession'])
            s_features = [
                'sex_M',
                'sex_F',
                'sex_Z',
                'race_A',
                'race_B',
                'race_I',
                'race_P',
                'race_Q',
                'race_U',
                'race_W',
                'race_Z',
            ]
            y_features = ['possession']
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    return x_features, s_features, y_features


def _split_features(df, data_name, feature_split):
    x_features, s_features, y_features = _split_feature_names(
        df, data_name, feature_split)

    # allowing to go back from one hot encoded features to categorical features
    if data_name=="adult":
        SORTED_FEATURES_NAMES = [
                'age',
                'education-num',
                'capital-gain',
                'capital-loss',
                'hours-per-week',
                'workclass',
                'education',
                'marital-status',
                'occupation',
                'relationship',
                'race',
                'sex',
                'native-country',
                'salary'   
            ]

    features = OrderedDict()
    for i in range(len(SORTED_FEATURES_NAMES)):
        features[SORTED_FEATURES_NAMES[i]] = [not re.match(SORTED_FEATURES_NAMES[i],values)==None for values in x_features]

    # fixing the education to not count education-num
    features['education'][1] = False
    x = df[x_features].values.astype(float)
    s = df[s_features].values.astype(float)
    y = df[y_features].values.astype(float)

    return x, s, y, features


def scale(scaler_class, train, valid, test):
    if scaler_class is None:
        return [train, valid, test]

    scaler = scaler_class()
    scalerobj = scaler.fit(np.concatenate((np.concatenate((train,valid),axis=0),test),axis=0))
    train_scaled = scalerobj.transform(train)
    valid_scaled = scalerobj.transform(valid)
    test_scaled = scalerobj.transform(test)
    return [train_scaled, valid_scaled, test_scaled]


def load_dataset(train_path, test_path, validation_size, random_state,
                 data_name, feature_split, remake_test=False, test_size=None,
                 input_scaler=StandardScaler, sensitive_scaler=None):
    df_train_raw = pd.read_csv(train_path, engine='c')
    df_test_raw = pd.read_csv(test_path)

    if remake_test:
        if test_size is None:
            test_size = df_test_raw.shape[0]

        df_full = pd.concat([df_train_raw, df_test_raw])
        df_full_shuffled = df_full.sample(frac=1, random_state=random_state)

        df_train_valid = df_full_shuffled[:-test_size]
        df_test = df_full_shuffled[-test_size:]

    else:
        if test_size is not None:
            raise ValueError("Changing test size is only possible "
                             "if remake_test is True.")

        df_train_valid = df_train_raw.sample(frac=1, random_state=random_state)
        df_test = df_test_raw

    df_train = df_train_valid[:-validation_size]
    df_valid = df_train_valid[-validation_size:]

    x_train, s_train, y_train, cat_features = _split_features(df_train, data_name, 
                                                feature_split)
    x_valid, s_valid, y_valid,_ = _split_features(df_valid, data_name, 
                                                feature_split)
    x_test, s_test, y_test,_ = _split_features(df_test, data_name, 
                                             feature_split)

    x_train, x_valid, x_test = scale(
        input_scaler, x_train, x_valid, x_test)

    s_train, s_valid, s_test = scale(
        sensitive_scaler, s_train, s_valid, s_test)

    data_train = x_train, s_train, y_train
    data_valid = x_valid, s_valid, y_valid
    data_test = x_test, s_test, y_test

    return data_train, data_valid, data_test, cat_features

def main():
    train_path = 'preprocessed/train.csv'
    test_path = 'preprocessed/test.csv'
    RANDOM_SEEDS = [87656123, 741246123, 292461935, 502217591, 9327935, 2147631, 2010588, 5171154, 6624906, 5136170]
    validation_size = 2000
    test_size = 15000
    data_name = 'adult'
    feature_split = 'sex_salary'
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
    s_column_names = ["sensitive"]
    y_column_names = ["label"]
    for i in range(10):
        makedirs(f'splits/{i}/original/', exist_ok=True)
        makedirs(f'splits/{i}/results/', exist_ok=True)
        train_df, _, test_df, _ = load_dataset(train_path, test_path, validation_size,
                                               RANDOM_SEEDS[i], data_name, feature_split,
                                               remake_test=True, test_size=test_size)
        x_train, s_train, y_train = train_df
        x_test, s_test, y_test = test_df
        pd.DataFrame(x_train, columns=x_column_names).to_csv(f'splits/{i}/original/x_train.csv', index=False)
        pd.DataFrame(s_train, columns=s_column_names).to_csv(f'splits/{i}/original/s_train.csv', index=False)
        pd.DataFrame(y_train, columns=y_column_names).to_csv(f'splits/{i}/original/y_train.csv', index=False)
        pd.DataFrame(x_test, columns=x_column_names).to_csv(f'splits/{i}/original/x_test.csv', index=False)
        pd.DataFrame(y_test, columns=y_column_names).to_csv(f'splits/{i}/original/y_test.csv', index=False)
        pd.DataFrame(s_test, columns=s_column_names).to_csv(f'splits/{i}/original/s_test.csv', index=False)


if __name__ == "__main__":
    main()