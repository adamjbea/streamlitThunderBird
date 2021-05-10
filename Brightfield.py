#@title Brightfield Analysis
###############################################################################
def Brightfield_Controller(dire):
  uploaded = []
  return_list = []
  uploaded,filenames = all_imgs_directory(dire)
  filenames = [filename.replace(dire + '/', '') for filename in filenames]
  if filenames:
      print(filenames)
      img_bw = []

      for key in uploaded:
        if key == None:
          break
        img = np.asarray(key)  
        color = Color_Detection(img)
        if color == 'grey':
          img_bw.append(img)
      
      if len(img_bw) > 0:
        try:
          print("Running Brightfield Analysis")
          analysisdata, anlayzed_imgs_bf = Brightfield_Analysis(img_bw)
          i = 0
          if Output_Folder:
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
              return_data = csvappend + picturedata
              return_list.append(return_data)
            return return_list
        except:
          print("Error")
    
  else:
    print("I SEE NO IMAGES HERE!\nPlease check the folder and if correct restart application and try again.")
  
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

    print("Number of brightfield images in folder: " + str(len(img_bw)))
    
    for img_set in range(len(img_bw)): 
      BAD_IMAGE = False
      processed_img = Incoming_Image_Processing(img_bw[img_set])
      accuracy_img_copy = processed_img.copy()
      bead_img_copy = processed_img.copy()

      circles_grey, zoom = Circle_Finder(processed_img, 'grey')
      
      if circles_grey is not None and zoom != "ERROR":

        bead_load = ""
        bead_count = ""
        nobead_count = ""
        
        if zoom == "large":
          accurate_circles = Circle_Finder_Improve(circles_grey,processed_img)
          bead_count = 0 
          nobead_count = 0
          bead_count, nobead_count, circles_with_beads = Bead_Counting(accurate_circles, bead_img_copy)
          total_drops = bead_count + nobead_count
          if total_drops == 0:
            BAD_IMAGE = True
            total_drops = 1
            bead_count = 1
          else:
            bead_load = str(round((bead_count/total_drops)*100))+"%"
        else:
          accurate_circles = circles_grey[0]
          
        key_points = Blob_Detect(processed_img, zoom)
        kp_read = Readable_Key_Points(key_points)
        
        filter = len(accurate_circles)
        if filter>50 and BAD_IMAGE == False:

          circles_img = Draw_Circles(processed_img, accurate_circles,circles_with_beads)   
          if Output_Browser:
            print("\n\n#################################################\nNEXT IMAGE BELOW\n")
            print("Image size: " + str(processed_img.shape))
            print("ZOOM OF IMAGE IS: " + zoom + " size")
            print("\n") 

            print(" - HOUGH DROP DETECTION - ")

            if zoom == "large":
              print(" - BEAD COUNTING - ")
              print("\nTotal Drops: " + str(total_drops))
              print("Bead: " + str(bead_count))
              print("Nobead: " + str(nobead_count))
              print("BEAD LOADING: " + bead_load)
              print("\n") 

          return_data = Analytics_Display(accurate_circles,kp_read,zoom)
          img_blobs = Draw_Blobs(circles_img, key_points)
          analyzed_img.append(img_blobs)
          
          if Output_Browser:   
            print("\n")
            print("Radius from Hough")
            Radius_Histogram(circles_grey[0])
            print("\n")
            
            print("Detection Image")
            plt.figure(figsize=fig_size)
            plt.imshow(img_blobs)
            plt.show()

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
    print("IMAGES ARE CORRUPTED AND NOTHING WAS DONE")
  return data_collection, analyzed_img