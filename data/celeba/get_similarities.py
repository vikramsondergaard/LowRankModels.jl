import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.metrics import mean_squared_error

def get_similarities(seed):                                         
    n_images = 540                                                  
    rmse = dict()
    path = os.path.join('/Users', 'vikramsondergaard', 'honours',
                        'LowRankModels.jl', 'data', 'celeba', 'splits', 
                        str(seed), 'results', 'images')
    for i in range(n_images):
        print(f'{100 * i / n_images}% complete...')
        for j in range(i + 1, n_images):
            imgpath1 = os.path.join(path, os.listdir(path)[i])
            imgpath2 = os.path.join(path, os.listdir(path)[j])
            img1 = Image.open(imgpath1)
            img2 = Image.open(imgpath2)
            mse = mean_squared_error(np.array(img1), np.array(img2))
            rmse[f'({os.listdir(path)[i]}, {os.listdir(path)[j]})'] = mse
    return rmse


def view_images(seed, image1, image2):
    df = pd.read_csv(f'splits/{seed}/original/x_train.csv')
    names = df['Name']
    attractive = pd.read_csv('attractive.csv')
    img1name = names[image1]
    img2name = names[image2]
    img1attractive = attractive.iloc[int(img1name[:-4]) - 1, 1]
    img2attractive = attractive.iloc[int(img2name[:-4]) - 1, 1]
    if img1attractive != img2attractive:
    # print(f'Image {image1} is attractive?: {img1attractive}')
    # print(f'Image {image2} is attractive?: {img2attractive}')
        orig_img1 = Image.open(os.path.join('img_align_celeba', str(img1name)))
        orig_img2 = Image.open(os.path.join('img_align_celeba', str(img2name)))
        results_path = os.path.join('/Users', 'vikramsondergaard', 'honours',
                                    'LowRankModels.jl', 'data', 'celeba', 'splits',
                                    str(seed), 'results', 'images')
        proj_img1 = Image.open(os.path.join(results_path, 
                                            f'split_{seed}_img{image1}.png'))
        proj_img2 = Image.open(os.path.join(results_path, 
                                            f'split_{seed}_img{image2}.png'))
        imgpath = os.path.join('for_paper', f'imgs_{image1}_{image2}')
        attrpath = os.path.join(imgpath, 'attractive')
        unattrpath = os.path.join(imgpath, 'unattractive')
        os.makedirs(attrpath, exist_ok=True)
        os.makedirs(unattrpath, exist_ok=True)
        img1path = os.path.join(imgpath, attstr(img1attractive))
        img2path = os.path.join(imgpath, attstr(img2attractive))
        orig_img1.save(fp=os.path.join(img1path, f'orig_img_{img1name[:-4]}.png'))
        orig_img2.save(fp=os.path.join(img2path, f'orig_img_{img2name[:-4]}.png'))
        proj_img1.save(fp=os.path.join(img1path, f'proj_img_{img1name[:-4]}.png'))
        proj_img2.save(fp=os.path.join(img2path, f'proj_img_{img2name[:-4]}.png'))



def view_images_from_string(s):
    imgs = s.split(', ')
    img1 = imgs[0]
    img2 = imgs[1]
    img1 = img1[1:] # remove the left bracket
    img2 = img2[:-1] # remove the right bracket
    seed = int(img1[6])
    image1 = int(img1[11:-4])
    image2 = int(img2[11:-4])
    view_images(seed, image1, image2)


def attstr(is_attractive):
    return 'attractive' if is_attractive else 'unattractive'