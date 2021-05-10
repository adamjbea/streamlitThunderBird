import statistics
import numpy as np

def Analytics_Display(circles, kp_read,zoom):
  radius_list = []
  for c in circles:
    radius_list.append(c[2])

  larger_med = []
  area_list = []
  for radi in radius_list:
    area = 3.14*radi**2
    if area > 1000:
      area_list.append(area)
  number = len(area_list)
  median = statistics.median(area_list)

  avg_r = round(Average_Radius(circles),2)
  std_r = round(STD_Radius(radius_list),2)
  area_std = round(np.std(area_list),2)
  
  if area_std < 700:
    median_factor = 2
    blob_factor = 1.8
  elif area_std > 1000:
    median_factor = 1.35
    blob_factor = 2
  else:
    median_factor = 1.55
    blob_factor = 1.9

  for area in area_list:
    if area > median_factor*median:
      larger_med.append(area)

  arealargermedia = sum(larger_med)
  areatotal = int(sum(area_list))

  num_blobs = len(kp_read)
  
  if zoom == "large":
    factor = .0225
  elif zoom == 'small':
    factor = .15
  else:
    factor = .5

  if num_blobs > 0:
    volume_mergers = [item[0]**blob_factor for item in kp_read]
  else:
    volume_mergers = [0]
  Num2xMed =  len(larger_med) + num_blobs
  Area2xMed = int(arealargermedia) + int(sum(volume_mergers))
  lessthan = 0
  for i in volume_mergers:
    if i > 1000:
      lessthan += i
  Area2xMedless10000 = Area2xMed - int(lessthan)
  emulsion_stability = round((Area2xMed/areatotal)*100)

  return_data = (['Brightfield', int(median*factor), int(area_std*factor), Num2xMed, Area2xMed, areatotal, emulsion_stability, "", "", number])

  #Display outputs for browser viewing
  #####################
  #Significant speed up if not performed
  print("\nTotal Drops: " + str(number))
  print("AVG Size: " + str(avg_r))
  print("STD: " + str(std_r))
  print("\n")
  print(" - MERGER DROP DETECTION - ")
  print("\n") 

  print("Number of Large Mergers: " + str(num_blobs))

  print("Area of mergers found: " + str(volume_mergers))
  

  print("Emulsions Stability; " + str(emulsion_stability))
  print("Number Blobs and Drops: " + str(num_blobs+number))

  #####################


  return return_data
  
###############################################################################
def Average_Radius(circles: list) -> float:

  sum_radius = 0
  for i in circles:
    sum_radius = i[2] + sum_radius

  return sum_radius/len(circles)

###############################################################################
def STD_Radius(radius_list: list) -> float:

  std = np.std(radius_list)

  return std

###############################################################################
def Sort_Circle(circles: list, selection: str) -> list:
  
  if selection == 'x':
    circles = circles[circles[:,0].argsort()]
  if selection == 'y':
    circles = circles[circles[:,1].argsort()]
  if selection == 'r':
    circles = circles[circles[:,2].argsort()]

  return circles

###############################################################################
class MSImage():
    """Lightweight wrapper for handling image to matrix transforms. No setters,
    main point of class is to remember image dimensions despite transforms."""
    
    def __init__(self, img):
        """Assume color channel interleave that holds true for this set."""
        self.img = img
        self.dims = np.shape(img)
        self.mat = np.reshape(img, (self.dims[0] * self.dims[1], self.dims[2]))

    @property
    def matrix(self):
        return self.mat
        
    @property
    def image(self):
        return self.img
    
    def to_flat_img(self, derived):
        """"Use dims property to reshape a derived matrix back into image form when
        derived image would only have one band."""
        return np.reshape(derived, (self.dims[0], self.dims[1]))
    
    def to_matched_img(self, derived):
        """"Use dims property to reshape a derived matrix back into image form."""
        return np.reshape(derived, (self.dims[0], self.dims[1], self.dims[2]))

###############################################################################
def BNormalize(mat):
    """much faster brightness normalization, since it's all vectorized"""
    bnorm = np.zeros_like(mat, dtype=np.float32)
    maxes = np.max(mat, axis=1)
    bnorm = mat / np.vstack((maxes, maxes, maxes)).T
    return bnorm