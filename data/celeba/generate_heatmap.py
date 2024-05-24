import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

def convert_to_grayscale():
    for imgpair in os.listdir('for_paper'):
        for attractive in ['attractive', 'unattractive']:
            path = os.path.join('for_paper', imgpair, attractive)
            for i in os.listdir(path):
                if i.startswith('orig'):
                    img = Image.open(os.path.join(path, i)).convert('L')
                    resized_image = img.resize((64, 64))
                    resized_image.save(os.path.join(path, 'rescaled' + i[4:]))


def generate_heatmap():
    for imgpair in os.listdir('celebfaces'):
        for attractive in ['attractive', 'unattractive']:
            path = os.path.join('celebfaces', imgpair, attractive)
            projected_image = None
            rescaled_image = None
            for i in os.listdir(path):
                if i.startswith('rescaled'):
                    rescaled_image = Image.open(os.path.join(path, i))
                elif i.startswith('proj'):
                    projected_image = Image.open(os.path.join(path, i))
            diff = np.abs(np.array(projected_image) - np.array(rescaled_image))
            fig, ax = plt.subplots()
            im = ax.imshow(diff)
            cbar = ax.figure.colorbar(im, ax=ax)
            cbarlabel = 'Projection distance'
            cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
            fig.tight_layout()
            plt.savefig(os.path.join(path, 'heatmap.png'))
            plt.close()


def generate_region_distance():
    region_df = pd.read_csv('Anno/list_landmarks_resampled_celeba.csv', index_col=0)
    data = []
    for imgpair in os.listdir('for_paper'):
        for attractive in ['attractive', 'unattractive']:
            path = os.path.join('for_paper', imgpair, attractive)
            projected_image = None
            rescaled_image = None
            imgname = None
            for i in os.listdir(path):
                if i.startswith('rescaled'):
                    rescaled_image = Image.open(os.path.join(path, i))
                elif i.startswith('proj'):
                    projected_image = Image.open(os.path.join(path, i))
                imgname = i.split('_')[-1][:-3] + 'jpg'
            diff = np.abs(np.array(projected_image) - np.array(rescaled_image))
            regions = region_df.loc[imgname, :]
            eyes = regions[0:4]
            nose = regions[4:6]
            mouth = regions[6:]
            datapoint = [imgname, attractive == 'attractive']
            eye_ylower = min(eyes[1], eyes[3])
            eye_yupper = max(eyes[1], eyes[3])
            eye_xlower = min(eyes[0], eyes[2])
            eye_xupper = max(eyes[0], eyes[2])
            eyes_rows = diff[eye_ylower:eye_yupper+1]
            eyes_diff = [row[eye_xlower:eye_xupper+1] for row in eyes_rows]
            print(f'eyes_diff: {eyes_diff}')
            datapoint.append(np.mean(eyes_diff))
            nose_pixel = diff[nose[1]][nose[0]]
            print(f'nose_pixel: {nose_pixel}')
            datapoint.append(nose_pixel)
            mouth_ylower = min(mouth[1], mouth[3])
            mouth_yupper = max(mouth[1], mouth[3])
            mouth_xlower = min(mouth[0], mouth[2])
            mouth_xupper = max(mouth[0], mouth[2])
            mouth_rows = diff[mouth_ylower:mouth_yupper+1]
            mouth_diff = [row[mouth_xlower:(mouth_xupper+1)] for row in mouth_rows]
            print(f'mouth_diff: {mouth_diff}')
            datapoint.append(np.mean(mouth_diff))
            data.append(datapoint)
    columns = ['Name', 'Attractive?', 'Mean Eye Projection Distance',
               'Nose Projection Distance', 
               'Mean Mouth Projection Distance']
    df = pd.DataFrame(np.array(data), columns=columns)
    df.to_csv('projection_distances.csv', index=False)



# if __name__ == "__main__":
#     convert_to_grayscale()