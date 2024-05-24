import pandas as pd
import numpy as np
import argparse
import os

from PIL import Image
from sklearn.preprocessing import StandardScaler


def parse_args():
    """
    Parse the arguments given to this script. These arguments are
    - the path to the image data
    - the path to the attribute data (of whether each celebrity
      is deemed attractive)
    - the number of required samples of attractive celebrities
    - the number of required samples of unattractive celebrities
    - the resampling for x (optional, defaults to 64)
    - the resampling for y (optional, defaults to 64)
    - the random seed (optional, defaults to 0)

    :return: the parsed arguments
    """
    print("Starting parse_args()...")
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath')
    parser.add_argument('attractivepath')
    # My current experiments use 405 attractive samples
    parser.add_argument('n_attractive')
    # My current experiments use 405 unattractive samples
    parser.add_argument('n_unattractive')
    parser.add_argument('-x', '--samplex', default = 64)
    parser.add_argument('-y', '--sampley', default = 64)
    parser.add_argument('-s', '--seed', default = 0)
    args = parser.parse_args()
    print("Finished parse_args().")
    return args


def get_datapoints(datapath, attractivepath, n_attractive, n_unattractive, seed):
    """
    Get the number of desired datapoints, separated by attractive and
    unattractive celebrities.

    :param datapath: the path to the image data
    :param attractivepath: the path to the attribute data (of whether each celebrity
      is deemed attractive)
    :param n_attractive: the number of required samples of attractive celebrities
    :param n_unattractive: the number of required samples of unattractive celebrities
    :param seed: the random seed

    :return: the resulting samples of attractive and unattractive celebrities
    """
    print("Starting get_datapoints()...")
    df = pd.read_csv(datapath)
    attractive = pd.read_csv(attractivepath)
    full_df = df.join(attractive[['Name', 'Attractive']])
    attractive_df = full_df[full_df['Attractive']]
    unattractive_df = full_df[full_df['Attractive'] == False]
    attractive_samples = attractive_df.sample(n=n_attractive, random_state=seed)
    unattractive_samples = unattractive_df.sample(n=n_unattractive, random_state=seed)
    sampled_df = pd.concat([attractive_samples, unattractive_samples])
    shuffled_df = sampled_df.sample(frac=1.0, random_state=888)
    print("Finished get_datapoints().")
    return shuffled_df


def resample_images(df, sample_x, sample_y):
    """
    Resamples the images in a given dataset.

    :param df: the image dataset to resample
    :param sample_x: the resampling size along the x-axis
    :param sample_y: the resampling size along the y-axis

    :return: the resampled image data
    """
    print("Starting resample_images()...")
    resampled_data = []
    dim_x, dim_y = eval(df.columns[-1])
    for i in range(df.shape[0]):
        # Have to add 1 to dim_x and dim_y because they're 0-indexed in columns
        img_datapoint = np.array(df.iloc[i, :]).reshape(dim_x + 1, dim_y + 1)
        img = Image.fromarray(img_datapoint, 'L')
        resized_img = img.resize((sample_x, sample_y))
        resampled_data.append(np.array(resized_img).flatten())
    columns = [f'({x}, {y})' for y in range(sample_y) for x in range(sample_x)]
    out_df = pd.DataFrame(np.array(resampled_data), columns=columns, index=df.index)
    print("Finished resample_images().")
    return out_df


def get_images(img_dir, attr_path, n_attractive, n_unattractive, sample_x, sample_y, seed):
    attr = pd.read_csv(attr_path)
    attractive = attr[attr['Attractive']]
    unattractive = attr[attr['Attractive'] == False]
    attractive_imgs = attractive['Name'].sample(n=n_attractive, random_state=seed)
    unattractive_imgs = unattractive['Name'].sample(n=n_unattractive, random_state=seed)
    data = []
    for img in attractive_imgs:
        image = Image.open(os.path.join(img_dir, img)).convert('L')
        resized = image.resize((sample_x, sample_y))
        flattened = list(np.array(resized).flatten())
        flattened.append(True)
        flattened.append(img)
        data.append(flattened)
    for img in unattractive_imgs:
        image = Image.open(os.path.join(img_dir, img)).convert('L')
        resized = image.resize((sample_x, sample_y))
        flattened = list(np.array(resized).flatten())
        flattened.append(False)
        flattened.append(img)
        data.append(flattened)
    columns = [f"({x}, {y})" for y in range(sample_y) for x in range(sample_x)]
    columns.extend(['Attractive', 'Name'])
    return pd.DataFrame(data, columns=columns)


# def main():
#     print("Starting main()...")
#     args = parse_args()
#     df = get_datapoints(args.datapath, args.attractivepath, int(args.n_attractive), int(args.n_unattractive), int(args.seed))
#     img_df = df.loc[:, df.columns != 'Attractive']
#     resampled_df = resample_images(img_df, args.samplex, args.sampley)
#     full_df = resampled_df.join(df['Attractive'])
#     full_df.to_csv('preprocessed_img_data.csv', index=False)
#     print("Done!")

def main():
    RANDOM_SEEDS = [87656123, 741246123, 292461935, 502217591, 9327935, 2147631, 2010588, 5171154, 6624906, 5136170]
    args = parse_args()
    df = get_images(args.datapath, args.attractivepath, int(args.n_attractive), int(args.n_unattractive), int(args.samplex), int(args.sampley), RANDOM_SEEDS[int(args.seed)])
    # Have to do this because the image values are centred on 0 for some reason
    # scaled_df = df.loc[:, df.columns != 'Attractive'].applymap(lambda x : x + 128)
    # df = scaled_df.join(df['Attractive'])
    # scaler = StandardScaler()
    # scaler.fit_transform(df)
    train_split = int(2 * (int(args.n_attractive) + int(args.n_unattractive)) / 3)
    df = df.sample(frac=1., random_state=888)
    df_train = df.iloc[:train_split, :]
    df_test = df.iloc[train_split:, ]
    x_train = df_train.loc[:, df_train.columns != 'Attractive']
    s_train = df_train.loc[:, 'Attractive']
    y_train = s_train
    x_test = df_test.loc[:, df_test.columns != 'Attractive']
    s_test = df_test.loc[:, 'Attractive']
    y_test = s_test
    x_train.to_csv(f'splits/{args.seed}/original/x_train.csv', index=False)
    y_train.to_csv(f'splits/{args.seed}/original/y_train.csv', index=False)
    s_train.to_csv(f'splits/{args.seed}/original/s_train.csv', index=False)
    x_test.to_csv(f'splits/{args.seed}/original/x_test.csv', index=False)
    y_test.to_csv(f'splits/{args.seed}/original/y_test.csv', index=False)
    s_test.to_csv(f'splits/{args.seed}/original/s_test.csv', index=False)


if __name__ == "__main__":
    main()
