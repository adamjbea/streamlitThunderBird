import Tools
import Detection
import numpy as np
import cv2
import matplotlib as plt
import Analyze
import Visualization
import streamlit as st
import Beadcount
from matplotlib import pyplot as plt
import pandas as pd

def Brightfield_Controller(dire):
  df_keys = ["Image Name", 
              "Image Location", 
              "Processed Image Location", 
              "Analysis Type", 
              "Med Area", 
              "Area STD", 
              "#>2x Med", 
              "Area>2x Med", 
              "Total Area", 
              "Emulsion Stability", 
              " ", 
              " ", 
              "Total Drops", 
              "%Bead Loading", 
              "Number of Beads", 
              "Number of No Beads"]
  uploaded = []
  return_list = []
  uploaded,filenames = Tools.all_imgs_directory(dire)
  filenames = [filename.replace(dire + '/', '') for filename in filenames]
  if filenames:
      #st.write(filenames)
      img_bw = []

      for key in uploaded:
        if key == None:
          break
        img = np.asarray(key)  
        color = Detection.Color_Detection(img)
        if color == 'grey':
          img_bw.append(img)
      
      if len(img_bw) > 0:
        try:
          st.write("Running Brightfield Analysis")
          analysisdata, anlayzed_imgs_bf = Brightfield_Analysis(img_bw)
          i = 0
          for picturedata in analysisdata:     
            filename = filenames[i]
            img_name = filename.replace(dire, '')
            filename.replace('filename', '')
            save_name = dire + "/Analyzed_BF_" + img_name
            if len(analysisdata) == len(anlayzed_imgs_bf):
              image_to_write = cv2.cvtColor(anlayzed_imgs_bf[i], cv2.COLOR_RGB2BGR)
              cv2.imwrite(save_name, image_to_write) 
            csvappend = [img_name, dire, save_name]
            i += 1
            temp_data = csvappend + picturedata
            df = pd.DataFrame()
            for count, data in enumerate(temp_data):
              if count == 16:
                break
              else:
                st.write("Count: ", count)
                st.write("Data: ", data)
                df[df_keys[count]] = "testing"
                st.write(df[df_keys[count]])
            return_list = []
            for x in range(3):
              st.write(temp_data[x])
              return_list.append(temp_data[x])
            return_list.append(df)
          return return_list
        except Exception as e:
          st.write("Error:", e)
    
  else:
    st.write("I SEE NO IMAGES HERE!\nPlease check the folder and if correct restart application and try again.")
  
# Handles all brightficeld actions
# Calls all analysis and detection functions
# Also is in charge of formatting and presenting information to user
###############################################################################
def Brightfield_Analysis(img_bw):
  if img_bw is not None:
    analyzed_img = []
    accurate_circles = []
    data_collection = []
    volume_mergers = []
    circles_with_beads = []

    st.write("Number of brightfield images in folder: " + str(len(img_bw)))
    
    for img_set in range(len(img_bw)): 
      BAD_IMAGE = False
      processed_img = Tools.Incoming_Image_Processing(img_bw[img_set])
      accuracy_img_copy = processed_img.copy()
      bead_img_copy = processed_img.copy()

      circles_grey, zoom = Detection.Circle_Finder(processed_img, 'grey')
      
      if circles_grey is not None and zoom != "ERROR":

        bead_load = ""
        bead_count = ""
        nobead_count = ""
        
        if zoom == "large":
          accurate_circles = Detection.Circle_Finder_Improve(circles_grey,processed_img)
          bead_count = 0 
          nobead_count = 0
          bead_count, nobead_count, circles_with_beads = Beadcount.Bead_Counting(accurate_circles, bead_img_copy)
          total_drops = bead_count + nobead_count
          if total_drops == 0:
            BAD_IMAGE = True
            total_drops = 1
            bead_count = 1
          else:
            bead_load = str(round((bead_count/total_drops)*100))+"%"
        else:
          accurate_circles = circles_grey[0]
          
        key_points = Detection.Blob_Detect(processed_img, zoom)
        kp_read = Tools.Readable_Key_Points(key_points)
        
        filter = len(accurate_circles)
        if filter>50 and BAD_IMAGE == False:

          circles_img = Visualization.Draw_Circles(processed_img, accurate_circles,circles_with_beads)
          """ 
          st.write("#################################################")
          st.write("NEXT IMAGE BELOW")
          st.write("Image size: " + str(processed_img.shape))
          st.write("ZOOM OF IMAGE IS: " + zoom + " size")
          st.write(" - HOUGH DROP DETECTION - ")
          if zoom == "large":
            st.write(" - BEAD COUNTING - ")
            st.write("Total Drops: ", str(total_drops))
            st.write("Bead: ", str(bead_count))
            st.write("Nobead: ", str(nobead_count))
            st.write("BEAD LOADING: ", bead_load)
          """

          return_data = Analyze.Analytics_Display(accurate_circles,kp_read,zoom)
          img_blobs = Visualization.Draw_Blobs(circles_img, key_points)
          analyzed_img.append(img_blobs)
          
          st.write("Radius from Hough")
          Visualization.Radius_Histogram(circles_grey[0])
          
          """
          st.write("Detection Image")
          plt.figure(figsize= (10, 10))
          plt.imshow(img_blobs)
          plt.show()
          #st.image(img_blobs)
          """
          return_data.append(bead_load)
          return_data.append(bead_count)
          return_data.append(nobead_count)
          data_collection.append(return_data)
        else:
          return_data = (['Brightfield', "", "", "", "", "", "", "", "", ""])
          return_data.append(bead_load)
          return_data.append(bead_count)
          return_data.append(nobead_count)
          return_data.append("THIS IMAGE WAS PARTLY PROCESSED DUE TO POOR QUALITY")
          data_collection.append(return_data)

      else:
        data_collection.append(["COULD NOT DETECT ANYTHING IN THIS IMAGE"])
        analyzed_img.append(processed_img)
  else:
    st.write("IMAGES ARE CORRUPTED AND NOTHING WAS DONE")
  return data_collection, analyzed_img