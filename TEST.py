import cv2
from tqdm import tqdm
import os
TRAIN_DIR ="AGE_DATA"
def label_img(img):
    word_label = img[0]
    print(word_label)
for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
