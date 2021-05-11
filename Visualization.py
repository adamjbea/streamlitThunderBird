from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import cv2
import numpy as np
import Visualization
from PIL import Image
import Analyze
import math 
import Tools
import streamlit as st

def Radius_Histogram(circles_grey, circles_blue = 'na', circles_green = 'na'):
  fig = plt.figure(figsize=(10, 10))
  if circles_blue == 'na':
    r_bw = circles_grey[:, [2]]
    sns.distplot(r_bw, rug=True, rug_kws={"color": "k"},
                    kde_kws={"color": "k", "lw": 4, "label": "Grey"},
                    hist_kws={"histtype": "step", "linewidth": .1,
                              "alpha": 1, "color": "k"})

  else:
    r_b = circles_blue[:, [2]]
    r_g = circles_green[:, [2]]

    sns.distplot(r_b, rug=False, rug_kws={"color": "b"},
                      kde_kws={"color": "b", "lw": 2, "label": "Blue"},
                      hist_kws={"histtype": "step", "linewidth": .1,
                                "alpha": 1, "color": "b"})
    sns.distplot(r_g, rug=False, rug_kws={"color": "g"},
                      kde_kws={"color": "g", "lw": 2, "label": "Green"},
                      hist_kws={"histtype": "step", "linewidth": .1,
                                "alpha": 1, "color": "g"})
    r_bw = circles_grey[:, [2]]
    sns.distplot(r_bw, rug=True, rug_kws={"color": "k"},
                    kde_kws={"color": "k", "lw": 4, "label": "Grey"},
                    hist_kws={"histtype": "step", "linewidth": .1,
                              "alpha": 1, "color": "k"})

  #st.pyplot(fig)

#Plot detected drops to visually determine skew of 3 images
###############################################################################
def Scatter_Images(circles_grey, circles_blue, circles_green):
  plt.figure(figsize=(10, 10))
  color_bw = "grey"
  x_bw = circles_grey[:, [1]]
  y_bw = circles_grey[:, [0]]
  r_bw = circles_grey[:, [2]]

  color_b = "blue"
  x_b = circles_blue[:, [1]]
  y_b = circles_blue[:, [0]]
  r_b = circles_blue[:, [2]]

  color_g = 'green'
  x_g = circles_green[:, [1]]
  y_g = circles_green[:, [0]]
  r_g = circles_green[:, [2]]

  plt.scatter(x_bw, y_bw, s=r_bw, c=color_bw, alpha=0.7)
  plt.scatter(x_b, y_b, s=r_b, c=color_b, alpha=0.5)
  plt.scatter(x_g, y_g, s=r_g, c=color_g, alpha=0.5)

  plt.title('Image Skew Plot')
  plt.xlabel('Pixels X')
  plt.ylabel('Pixels Y')

  plt.show()

###############################################################################
def Draw_Blobs(image: list, key_points: list) -> list:
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_key_points = cv2.drawKeypoints(image,
                                          key_points,
                                          np.array([]),
                                          (0, 255, 0),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return im_with_key_points

###############################################################################
def Draw_Circles(image: list, circles: list, circles_with_beads) -> list:
    j = 0
    for i in circles:
      if len(circles_with_beads) > 0:
        if circles_with_beads[j]:
          cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 5)
        else:
          cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 5)
        j+=1
      else:
        cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), 5)
    return image

###############################################################################
def Draw_Circles_Inverse(image: list, circles: list) -> list:
    for i in circles[0, :]:
      cv2.circle(image, (i[0], i[1]), i[2], (255, 0, 0), -1)
    
    return image

#########################################################################  
def FluorescenceNormalize(blue, bw, green):

  # Make the images of uniform size
  image4 = Visualization.changeImageSize(2560, 1920, blue)
  image5 = Visualization.changeImageSize(2560, 1920, green)

  # alpha-blend the images with varying values of alpha
  alphaBlended2 = Image.blend(image4, image5, alpha=.5)

  msifirst = Analyze.MSImage(alphaBlended2)

  bnorm1 = Analyze.BNormalize(msifirst.matrix)
  bnorm_img1 = msifirst.to_matched_img(bnorm1)
  plt.figure(figsize=(10, 10))
  plt.imshow(bnorm_img1)
  plt.show()
  return bnorm_img1

###############################################################################
def Histogram_Grey(img_bw):

    plt.hist(img_bw.ravel(), 100, [20, 200], density=True, alpha=0.3, color="black")
    
    # Add labels
    plt.title('Drop Histogram')
    plt.xlabel('Brightness')
    plt.ylabel('Percentage of Color')

    no_fluor = mpatches.Patch(alpha = 0.3, color='black', label='No Florescence')

    plt.legend(handles=[no_fluor])
    plt.figure(figsize=(10, 10))
    plt.show()

######################################################################################
def Display_Pictures(print_pictures, range_list, name_list):
    range_list = Tools.Check_Size(print_pictures, range_list)

    size_wanted = range_list[1]-range_list[0]

    if size_wanted != 0:
        columns = int(math.sqrt(size_wanted))
        row = math.ceil(size_wanted / columns)

        j = 0
        plt.figure(figsize=(30,5))
        for i in range(10000)[range_list[0]:range_list[1]]:
            plt.subplot(columns, row, j + 1), plt.imshow(print_pictures[i], 'gray')
            plt.title(name_list[i])
            plt.xticks([]), plt.yticks([])
            j += 1
        plt.show()