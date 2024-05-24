import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv('projection_distances.csv')
    plt.rcParams['font.size'] = 22
    attractive_df = df[df['Attractive?']]
    unattractive_df = df[df['Attractive?'] == False]
    attractive = ['attractive', 'unattractive']
    for i, data in enumerate([attractive_df, unattractive_df]):
        fig = plt.figure(figsize=(12,8))
        plt.boxplot([np.array(data[f]) for f in df.columns if 'Projection Distance' in f])
        plt.xticks(ticks=[1,2,3], labels=['Eyes', 'Nose', 'Mouth'])
        plt.ylabel('Projection distance')
        plt.savefig(f'{attractive[i]}-distances.png')


if __name__ == "__main__":
    main()