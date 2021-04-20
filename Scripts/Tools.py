import glob
from PIL import Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

def Write_CSV(*args,custom_name): #For each column, pass a list where the first entry is the column's name.
    st.write(custom_name)
    rows = zip(*args)
    row_list = list(rows)
    data = np.array(row_list, dtype=object)
    df = pd.DataFrame.from_records(data)
    st.write(df)

###############################################################################
def all_imgs_directory(directory):
    image_list = []
    filenames = []
    print("Reading in Images..")
    for filename in glob.glob(directory + '/*.jpg'):
        if 'Analyzed' not in filename:
            im=Image.open(filename)
            filenames.append(filename)
            image_list.append(im)
    for filename in glob.glob(directory + '/*.PNG'):
        if 'Analyzed' not in filename:
            im=Image.open(filename)
            filenames.append(filename)
            im = im.convert('RGB')
            image_list.append(im)
    for filename in glob.glob(directory + '/*.JPG'):
        if 'Analyzed' not in filename:
            im=Image.open(filename)
            filenames.append(filename)
            im = im.convert('RGB')
            image_list.append(im)
    
    return image_list,filenames

###############################################################################
def Resize_Img_Grey(image):

    if (image.shape[0] == 960) or (image.shape[1] == 960):
        scale_percent = 200

        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)

        dimensions = (width, height)
        image_resize = cv2.resize(image, dimensions)
    
        return image_resize

    elif (image.shape[0] == 1920) or (image.shape[1] == 1920):
        return image

    else:
        scale_percent = 125

        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)

        dimensions = (width, height)
        image_resize = cv2.resize(image, dimensions)
    
        return image_resize 

###############################################################################
def Readable_Key_Points(key_points: list) -> list:
    readable_keypoint = []

    for x in range(len(key_points)):
        readable_keypoint.append(
            [round(key_points[x].size, 2), round(key_points[x].pt[0], 2), round(key_points[x].pt[1], 2)])
    return readable_keypoint

######################################################################################
def Brightfield_Norm(img,size):
    norm_img = np.zeros(size)
    normalized_img = cv2.normalize(img,  norm_img, 70, 255, cv2.NORM_MINMAX)
    if size == (1920,2560):
        avg_pixel = np.mean(normalized_img)
        if avg_pixel < 180:
            normalized_img = cv2.normalize(img,  norm_img, 70, 300, cv2.NORM_MINMAX)
    return normalized_img

######################################################################################
def Incoming_Image_Processing(img):
    resized_img = Resize_Img_Grey(img)
    normalized_img = Brightfield_Norm(resized_img,(1920,2560))

    processed_img = normalized_img
    return processed_img

######################################################################################
def Debug_Image(img, data_bool = False):

    if data_bool:
        try:
            print("Array: " + str(img))
        except:
            print("Array Fail")
        try:
            print("Shape: " + str(img.shape))
        except:
            print("Shape Fail")
        try:
            print("Dtype: " + str(img.dtype))
        except:
            print("Dtype Fail")
        try:
            print("Mean: " + str(np.mean(img)))
        except:
            print("Mean Fail")
    try:
        fig_size = (10, 10)
        plt.figure(figsize=fig_size)
        plt.imshow(img)
        plt.show()
    except:
        print("Display Img Failed")
###############################################################################
# Function to change the image size
def Change_Image_Size(shape, 
                    image):
    maxWidth = shape[0]
    maxHeight = shape[1]
    #convert a numpy array over to a PIL  image
    #https://stackoverflow.com/questions/10965417/how-to-convert-a-numpy-array-to-pil-image-applying-matplotlib-colormap
    image = Image.fromarray(image.astype('uint8'), 'RGB')

    widthRatio  = maxWidth/image.size[0]
    heightRatio = maxHeight/image.size[1]

    newWidth    = int(widthRatio*image.size[0])
    newHeight   = int(heightRatio*image.size[1])

    img    = image.resize((newWidth, newHeight))
    newImage = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
    return newImage

###############################################################################
def Image_Data_Convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

###############################################################################
def Channel_Split(image):
    r_img = image[:, :, 0]
    g_img = image[:, :, 1]
    b_img = image[:, :, 2]

    return r_img, g_img, b_img

######################################################################################
def Check_Size(data, range_list):
    if range_list[1] != 0 and len(data) < range_list[1]:
        range_list[1] = len(data)
    return range_list
