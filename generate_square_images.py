import cv2
import os

# AI4Shipwrecks helper script
# University of Michigan, Field Robotics Group
# Written by Anja Sheppard, 2023

# generate the square images for training/testing baselines

STRIDE = 100
RESIZE = 1024 # pixels h x w
IMG_DIR = '/path/to/images'
LBL_DIR = '/path/to/lbls'

# for images
for subdirs, dirs, files in os.walk(IMG_DIR):
    for file in files:
        if file.endswith('.png'):
            # for each scan in each survey: now compute the sliding window and get the crops
            img = cv2.imread(subdirs + '/' + file)
            height, width = img.shape[0], img.shape[1]
            # first, we need to pad the image so that it is equally divisible by the stride
            if height < 0.25 * width: continue # skip the image if the height is too short
            if height > width:
                pad_amount = STRIDE - ((height - width) % STRIDE)
            else:
                pad_amount = width - height
            img_padded = cv2.copyMakeBorder(img, 0, pad_amount, 0, 0, cv2.BORDER_CONSTANT, value=0)
            height_padded, width_padded = img_padded.shape[0], img_padded.shape[1]
            # now we cut the long image into multiple (height - width) / STRIDE + 1 images
            square_img_save_path = '/frog-drive/TBNMS/square_images_1024/'
            for i in range(int((height_padded - width_padded) / STRIDE) + 1):
                square_img = img_padded[100 * i:width + 100 * i]
                resized_square_img = cv2.resize(square_img, (RESIZE, RESIZE))
                cv2.imwrite(square_img_save_path + file[:-4] + '_' + str(i) + '.png', resized_square_img)

# for labels
for subdirs, dirs, files in os.walk(LBL_DIR):
    for file in files:
        if file.endswith('.png'):
            # for each scan in each survey: now compute the sliding window and get the crops
            label = cv2.imread(subdirs + '/' + file)
            height, width = label.shape[0], label.shape[1]
            # first, we need to pad the image so that it is equally divisible by the stride
            if height < 0.25 * width: continue # skip the image if the height is too short
            if height > width:
                pad_amount = STRIDE - ((height - width) % STRIDE)
            else:
                pad_amount = width - height
            label_padded = cv2.copyMakeBorder(label, 0, pad_amount, 0, 0, cv2.BORDER_CONSTANT, value=0)
            height_padded, width_padded = label_padded.shape[0], label_padded.shape[1]
            # now we cut the long image into multiple (height - width) / STRIDE + 1 images
            square_label_save_path = '/frog-drive/TBNMS/square_labels_1024/'
            for i in range(int((height_padded - width_padded) / STRIDE) + 1):
                square_label = label_padded[100 * i:width + 100 * i]
                resized_square_label = cv2.resize(square_label, (RESIZE, RESIZE))
                cv2.imwrite(square_label_save_path + file[:-4] + '_' + str(i) + '.png', resized_square_label)

print('Completed square images and labels.')
