import os
import cv2
import numpy as np
import pandas as pd
from scipy.misc import imresize
import scipy.io


dataset_path = './datasets/'

facescrub_path = os.path.join(dataset_path, 'facescrub')
#emoji_path = os.path.join(dataset_path, 'data', 'cars')

def shuffle_data(da, db):
    a_idx = range(len(da))
    np.random.shuffle( a_idx )

    b_idx = range(len(db))
    np.random.shuffle(b_idx)

    shuffled_da = np.array(da)[ np.array(a_idx) ]
    shuffled_db = np.array(db)[ np.array(b_idx) ]

    return shuffled_da, shuffled_db

def read_images( filenames, domain=None, image_size=64):

    images = []
    for fn in filenames:
        image = cv2.imread(fn)
        if image is None:
            continue

        if domain == 'A':
            kernel = np.ones((3,3), np.uint8)
            image = image[:, :256, :]
            image = 255. - image
            image = cv2.dilate( image, kernel, iterations=1 )
            image = 255. - image
        elif domain == 'B':
            image = image[:, 256:, :]

        image = cv2.resize(image, (image_size,image_size))
        image = image.astype(np.float32) / 255.
        image = image.transpose(2,0,1)
        images.append( image )

    images = np.stack( images )
    return images

def get_facescrub_files(test=False, n_test=200):
    actor_path = os.path.join(facescrub_path, 'actors', 'face' )
    actress_path = os.path.join( facescrub_path, 'actresses', 'face' )
    #pokemon_path = os.path.join(facescrub_path,'pokemon','face')
    emoji_path = os.path.join(facescrub_path,'emoji-data-master','img-apple-64')
    actor_files = map(lambda x: os.path.join( actor_path, x ), os.listdir( actor_path ) )
    actress_files = map(lambda x: os.path.join( actress_path, x ), os.listdir( actress_path ) )
   # pokemon_files = map(lambda x: os.path.join( pokemon_path, x ), os.listdir( pokemon_path ) )
    emoji_files = map(lambda x: os.path.join( emoji_path, x ), os.listdir( emoji_path ) )

    if test == False:
        return actor_files[:-n_test], actress_files[:-n_test]
    else:
        return actor_files[-n_test:], actress_files[-n_test:]


