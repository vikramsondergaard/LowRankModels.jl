import pandas as pd
import numpy as np

def main():
    data = []
    adult_df = pd.read_csv('splits/0/original/x_train.csv')
    for i in range(10):
        print(i)
        orig_path = f'splits/{i}/original/x_train.csv'
        ufdp_path = f'splits/{i}/results/x_train.csv'
        orig_df = np.array(pd.read_csv(orig_path))
        ufdp_df = np.array(pd.read_csv(ufdp_path))
        distances = np.linalg.norm(orig_df - ufdp_df, axis=0)
        data.append(distances)
    df = pd.DataFrame(data, columns=adult_df.columns)
    df.to_csv('distances.csv', index=False)


if __name__ == "__main__":
    main()