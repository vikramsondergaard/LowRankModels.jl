import pandas as pd
import numpy as np

def main():
    with open('Anno/list_landmarks_align_celeba.txt', 'r') as f:
        lines = f.readlines()[2:]
        data = [line.split() for line in lines]
        columns = ['name', 'lefteye_x', 'lefteye_y', 'righteye_x', 
                   'righteye_y', 'nose_x', 'nose_y', 'leftmouth_x', 
                   'leftmouth_y', 'rightmouth_x', 'rightmouth_y']
        df = pd.DataFrame(np.array(data), columns=columns)
        df.to_csv('Anno/list_landmarks_align_celeba.csv', index=False)


if __name__ == "__main__":
    main()