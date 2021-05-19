import cv2
import numpy as np
import helper.Tools as tools #type: ignore
from math import inf

def Circle_Finder_Improve(circles,img):

  #the lower this is the more it removes, max is 1 for no filtering
  FILTER_VALUE = 0.75

  indv_img_shape = (100,100)
  crop_detection_reduce = 45
  accurate_circles = []
  for c in circles[0, :]:
    if c[0] > 125 and c[0] < 2475:
      if c[1] > 125 and c[1] < 1825:
        average = 0
        if c is not None:
          img_copy = img.copy()
          img_clean = img.copy()
          c = c.astype(int)
          crop_bead_increase = c[2]*0.13

          cv2.circle(img_copy, (c[0], c[1]), c[2], (0, 255, 0), 10)

          crop_bead_increase = int(c[2] + crop_bead_increase)
        
          crop = img_copy[c[1]-crop_bead_increase:c[1]+crop_bead_increase, c[0]-crop_bead_increase:c[0]+crop_bead_increase]
          mask = np.zeros(crop.shape)
          mask = cv2.circle(mask, (crop_bead_increase, crop_bead_increase), crop_bead_increase, (255, 255, 255), -1)
          
          final_im = crop/mask
          final_im[final_im == inf] = 0

          crop_c = img_clean[c[1]-crop_bead_increase:c[1]+crop_bead_increase, c[0]-crop_bead_increase:c[0]+crop_bead_increase]
          final_im_c = crop_c/mask
          final_im_c[final_im_c == inf] = 0
          
          if np.count_nonzero(final_im) != 0:
            final_im = tools.Image_Data_Convert(final_im, 0, 255, np.uint8)  
            hsv = cv2.cvtColor(final_im, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0,200,0])
            upper_red = np.array([255,255,255])
            mask = cv2.inRange(hsv, lower_red, upper_red)
            res = cv2.bitwise_and(final_im_c,final_im_c, mask= mask)
            mean_outline = np.mean(res)
            average = res[np.nonzero(res)].mean()
            decision_img = final_im_c - res
            average_img = decision_img[np.nonzero(decision_img)].mean()
            decision_value = abs(average/average_img)

            if decision_value < FILTER_VALUE:
              accurate_circles.append(list(c))
            else:
              accurate_circles.append([0,0,0])

  return accurate_circles

###############################################################################
def Drops_Color(circles_grey,fluorescent_img):

  indv_img_shape = (100,100)
  crop_detection_reduce = 45
  color_avg_list = []
  for c in circles_grey[0, :]:
    average = 0
    if c is not None:
      img_copy = fluorescent_img.copy()
      img_clean = fluorescent_img.copy()
      c = c.astype(int)
      crop_bead_increase = c[2]*0.0

      cv2.circle(img_copy, (c[0], c[1]), c[2], (0, 255, 0), 10)

      crop_bead_increase = int(c[2] + crop_bead_increase)
    
      crop = img_copy[c[1]-crop_bead_increase:c[1]+crop_bead_increase, c[0]-crop_bead_increase:c[0]+crop_bead_increase]
      mask = np.zeros(crop.shape)
      mask = cv2.circle(mask, (crop_bead_increase, crop_bead_increase), crop_bead_increase, (255, 255, 255), -1)
      
      final_im = crop/mask
      final_im[final_im == inf] = 0

      crop_c = img_clean[c[1]-crop_bead_increase:c[1]+crop_bead_increase, c[0]-crop_bead_increase:c[0]+crop_bead_increase]
      final_im_c = crop_c/mask
      final_im_c[final_im_c == inf] = 0

      color_avg = final_im_c.mean(axis=0).mean(axis=0)
      color_avg_list.append(color_avg)

  return color_avg_list
  
#Look at a random pixel in an image and see what color it is
###############################################################################
def Color_Detection(cd_img: np.array):
  color_detected = "ERROR_Color_Detection"


  if (len(cd_img.shape)<3):
    return 'grey'

  color_avg = cd_img.mean(axis=0).mean(axis=0)
  color_std = round(np.std(color_avg),2)
  color_intensity = np.mean(color_avg)
  intensity_std = color_intensity/color_std
  offset = intensity_std*1.5

  if color_std == 0:
    color_detected = 'grey'

  elif intensity_std > 5:
    color_detected = 'bf_mixed'

  elif color_avg[0] > color_avg[1] + offset and color_avg [0] > color_avg[2] + offset:
    color_detected = 'red'

  elif color_avg[1] > color_avg[0] + offset and color_avg [1] > color_avg[2] + offset:
    color_detected = 'green'

  elif color_std < 15 and color_std > 2:
    color_detected = 'mixed'
    
  elif color_avg[2] > color_avg[1] + offset and color_avg [2] > color_avg[0] + offset:
    color_detected = 'blue'
  



  return color_detected

# blur and filter on intensite and detect circle for all but grey images
# grey images runs 3 seperate algorithms to detect the zoom currently being used and to classify it
###############################################################################
def Circle_Finder(img: np.array, color: str) ->list:

  zoom = "na"

  if color == 'green':
    img_g = img
    img_g = cv2.medianBlur(img_g, 5)
    lower_green= np.array([0, 72, 0])
    upper_green = np.array([10, 150, 10])
    mask = cv2.inRange(img_g, lower_green, upper_green)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=12, minRadius=35, maxRadius=60)
    
    #circles format = [[[x,y,radius],[x,y,radius],[x,y,radius]]] 
    return circles, zoom
  
  elif color == 'blue':
    img_b = img
    img_b = cv2.medianBlur(img_b, 5)
    lower_blue= np.array([0, 0, 90])
    upper_blue = np.array([10, 10, 110])
    mask = cv2.inRange(img_b, lower_blue, upper_blue)

    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 30, param1=50, param2=10, minRadius=35, maxRadius=55)
    
    # circles format = [[[x,y,radius],[x,y,radius],[x,y,radius]]]   
    return circles, zoom
  
  #####################
  # adaptive grey circle detector - runs multiple detections and chooses the best
  
  elif color == 'grey':

    zoom = "ERROR"

    if len(img.shape)<3:
      img_bw = img
    else:
      img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_bw = cv2.medianBlur(img_bw, 5)
    
    circles_a = [[0]]
    circles_b = [[0]]
    circles_c = [[0]]

    num_a = 0
    num_b = 0
    num_c = 0

    #HoughCicles is very finicky and will often crash if nothing is found 
    #try:
      #circles_a = cv2.HoughCircles(img_bw, cv2.HOUGH_GRADIENT, 1, 30,param1 = 60, param2 = 30, minRadius = 45, maxRadius = 70)
      #num_a = len(circles_a[0])
    #except:
      #pass
    try:
      circles_b = cv2.HoughCircles(img_bw, cv2.HOUGH_GRADIENT, 1, 30,param1 = 60, param2 = 30, minRadius = 30, maxRadius = 50)
      num_b = len(circles_b[0])
    except:
      pass
    try:
      circles_c = cv2.HoughCircles(img_bw, cv2.HOUGH_GRADIENT, 1, 50,param1 = 35, param2 = 40, minRadius = 70, maxRadius = 120)
      num_c = len(circles_c[0])
    except:
      pass

    #if (num_a > num_b) and (num_a > num_c):
      #circles = circles_a
      #zoom = "med"
    if (num_b > num_a) and (num_b > num_c):
      circles = circles_b
      zoom = "small"
    elif (num_c > num_a) and (num_c > num_b):
      circles = circles_c
      zoom = "large"
    else:
      circles = ["ERROR_ZOOM"]
      zoom = "ERROR"
    #circles format = [[[x,y,radius],[x,y,radius],[x,y,radius]]]  
    return circles, zoom

  elif color == 'ERROR':
    print("ERROR INCORRECT PICTURE TYPE LOADED!")
  else:
    circles = "Something is broken"

  return ["ERROR"], "ERROR"

#run one of 3 blob algorithms based on size 
###############################################################################
def Blob_Detect(im: list, zoom) -> list:

  # Setup SimpleBlobDetector parameters.
  params = cv2.SimpleBlobDetector_Params()

  params.filterByArea = True
  params.filterByCircularity = True
  params.filterByConvexity = True
  params.filterByInertia = False
  params.filterByColor = False
  params.minThreshold = 40
  params.maxThreshold = 225
  params.minCircularity = 0.60
  params.minConvexity = 0.25
  params.minInertiaRatio = 0.00

  if zoom == "small":
    params.minArea = 8000
    params.maxArea = 1000000

  if zoom == "med":
    params.minArea = 15000
    params.maxArea = 10000000

  if zoom == "large":
    params.minArea = 48000
    params.maxArea = 2500000

  # Create a detector with the parameters
  detector = cv2.SimpleBlobDetector_create(params)
  # Detect blobs.
  key_points = detector.detect(im)

  return key_points

#create a mask of the circle outlines and mean those values to determine accuracy of the overlap
#################################################|##############################
def Accuracy_Outline(img_circles, img_clean):

  hsv = cv2.cvtColor(img_circles, cv2.COLOR_BGR2HSV)
  lower_red = np.array([50,50,50])
  upper_red = np.array([255,255,255])
  mask = cv2.inRange(hsv, lower_red, upper_red)
  res = cv2.bitwise_and(img_clean,img_clean, mask= mask)
  mean_outline = np.mean(res)

  return mean_outline


###############################################################################
def get_contours(image):
    c_image = cv2.GaussianBlur(image,(5,5), 0)
    threshold = cv2.threshold(c_image,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    return contours

###############################################################################
def contour_analysis(img, contours, threshold, color, flag = False):
    contour_list = contours
    lst_display = []
    lst_intensities = []
    lst_underthresh = []
    
    for i in range(len(contour_list)):
      # Create a mask image that contains the contour filled in
      cimg = np.zeros_like(img)
      cv2.drawContours(cimg, contour_list, i, color=255, thickness=-1)
      if flag:
        tools.Debug_Image(cimg)
      # Access the image pixels and create a 1D numpy array then add to list
      pts = np.where(cimg == 255)
      avg = sum(img[pts[0], pts[1]]) / len(img[pts[0], pts[1]])
      lst_display.append(avg)
      if avg >= threshold:
        lst_intensities.append(avg)
      else:
        contour_list[i] = None



    return contour_list, lst_intensities,lst_display

###############################################################################
def Flour_Blob(image, color, threshold=0):
    #if image is not grey
    if len(image.shape)==3:
      image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 

    rough_contour_list = get_contours(image)
    pruned_contour_list, intensity_list,lst_display = contour_analysis(image, rough_contour_list, threshold, color)
    mask = np.zeros(image.shape, dtype=np.uint8)
    blobs = 0
    for c in pruned_contour_list:
        if c is None:
          continue
        area = cv2.contourArea(c)
        cv2.drawContours(mask, [c], -1, (36,255,12), cv2.FILLED)
        if area > 99: 
          blobs += 1
    bins = 15

    return mask, blobs, intensity_list,lst_display    