import numpy as np
import cv2
import Tools
from math import inf
def Bead_Counting(circles, img):

  bead_count = 0
  nobead_count = 0
  indv_img_shape = (100,100)
  crop_detection_reduce = 45
  img_copy = img.copy()
  img_list = []
  circles_with_beads = [False] * len(circles)

  circles = np.uint16(np.around(circles))

  for c in circles:
    c = c.astype(int)
    crop_bead_reduce = c[2]*0.1
    c[2] = c[2] - crop_bead_reduce
    crop = img_copy[c[1]-c[2]:c[1]+c[2], c[0]-c[2]:c[0]+c[2]]
    mask = np.zeros(crop.shape)
    mask = cv2.circle(mask, (c[2], c[2]), c[2], (255, 255, 255), -1)

    final_im = crop/mask

    final_im[final_im == inf] = 0
    img_list.append(final_im)

  i = 0
  for img in img_list[0:]:
    if img is not None:     
      shape = img.shape[0:2]
      if shape[0] == shape[1] and shape[0] != 0:
    
        img = Tools.Image_Data_Convert(img, 0, 255, np.uint8)    
        resized_img = Tools.Change_Image_Size(indv_img_shape,img)
        resized_img = Tools.Image_Data_Convert(resized_img, 0, 255, np.uint8)  

        if np.mean(img) > 200:
          normed_img = Tools.Brightfield_Norm(resized_img,indv_img_shape)
        else:
          normed_img = resized_img
        
        indv_copy = normed_img.copy()

        normed_img_new = normed_img.astype(np.uint8)
        normed_img_new = cv2.cvtColor(normed_img_new, cv2.COLOR_BGR2GRAY)
        thresholded = cv2.adaptiveThreshold(normed_img_new,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        mean_blur = cv2.medianBlur(thresholded, 5)
        mean_blur = cv2.bitwise_not(mean_blur)

        kernel = np.ones((2,2), np.uint8)
        erosion = cv2.erode(mean_blur,kernel,iterations = 1)
        dialation = cv2.dilate(erosion, kernel, iterations=1) 

        mask = np.zeros(dialation.shape)
        mask = cv2.circle(mask, (50, 50), crop_detection_reduce, (50, 50, 50), -1)
        mask = Tools.Image_Data_Convert(mask, 0, 255, np.uint8) 
        bitwise_and_img = cv2.bitwise_and(dialation, mask, mask = None) 

        circles = cv2.HoughCircles(bitwise_and_img, cv2.HOUGH_GRADIENT, 1, 30,param1 = 12, param2 = 12, minRadius = 25, maxRadius = 60)

        if circles is not None:
          circles_with_beads[i] = True
          bead_count += 1  
        else:
          nobead_count += 1

    i += 1
  return bead_count, nobead_count, circles_with_beads