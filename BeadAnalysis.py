import numpy as np

def Beads_Controller(dire):
  return_list = []
  #format: [[set1_identity, [[img1,color,name],[img2,color,name],[img3,color,name]]], [set2_identity, [[img1,color,name],[img2,color,name],[img3,color,name]]]]
  identifier_and_images = Flourecence_Image_Reader(dire)
  if identifier_and_images != []:
    data = Read_Decide_Analyze(identifier_and_images,dire)
  else:
    print("\nNothing in this folder: " + dire)

def Flourecence_Image_Reader(dire):
  uploaded = []
  return_list = []
  uploaded,filenames = all_imgs_directory(dire)
  filenames = [filename.replace(dire + '/', '') for filename in filenames]
  
  #format: [[set_identity, [[img1,color,name],[img2,color,name],[img3,color,name]]], [img_identity, [[img1,color,name],[img2,color,name],[img3,color,name]]]]
  identifier_and_images = []
  
  if filenames:
    print(filenames)

    image_iter_folder = 0
    for key in uploaded:
      if key == None:
        break
      img = np.asarray(key)  
      #####################
      #Algorithm: Image Set Generation
      #1.What color?
      color = Color_Detection(img)

      #Example image names, number is the color (d4 is normally brightfield)
      #Lubna-cy5_Slide 4_D_p00_0_A01f00d3
      #Lubna-cy5_Slide 4_D_p00_0_A01f00d4
      #Lubna-cy5_Slide 4_D_p00_0_A01f01d3
      #Lubna-cy5_Slide 4_D_p00_0_A01f01d4
      #2. What's its name?
      j = 0
      image_name = filenames[image_iter_folder].replace(dire, '')
      reversed_name = image_name[::-1]
      for letter in reversed_name:
        if letter == ' ' or letter == '_' or letter == '-':
          break      
        j += 1
      img_identity = reversed_name[5:j]

      #3. Do I have this in my list of sets?
      found_identity = False
      k = 0
      for img_set in identifier_and_images:
        if img_identity == img_set[0]:
          identifier_and_images[k][1].append([img,color,image_name])
          found_identity = True
        k+=1

      #4. I didn't find it in the set so make a new one
      if found_identity == False:
        identifier_and_images.append([img_identity,[[img,color,image_name]]])

      image_iter_folder += 1
  
  return identifier_and_images

  def Read_Decide_Analyze(identifier_and_images,Directory):

  final_data = pd.DataFrame()
  name_images = []
 # what type of image set is this? Then run analysis
  for image_set in identifier_and_images:
    print("\n\n#############################")
    #reverse back for human readability
    name = image_set[0][::-1]
    number_pictures = len(image_set[1])
    name_Mixed = None
    if number_pictures <= 4:
      empty_array = np.zeros((1,1))
      img_set = [empty_array,empty_array,empty_array,empty_array]
      
      for i in range(len(image_set[1])):
        color = image_set[1][i][1]
        image = image_set[1][i][0]
        
        if color == "grey":
          name_BF = image_set[1][i][2]
          img_set[0] = image

        elif color == "red":
          name_CY5 = image_set[1][i][2]
          img_set[1] = image

        elif color == "blue": 
          name_DAPI = image_set[1][i][2]
          img_set[2] = image

        elif color == "green":
          name_GFP = image_set[1][i][2]
          img_set[3] = image

        elif color == "bf_mixed":
          name_Mixed = image_set[1][i][2]

      if img_set[0].any() and img_set[1].any() and img_set[3].any(): 
        print("Set BW_R_G: " + str(name))
        imgs = load_images(dirpath = Directory+'/',BF = name_BF, GFP = name_GFP, CY5 = name_CY5)

      elif img_set[0].any() and img_set[3].any(): 
        print("Set BW_G: " + str(name))
        imgs = load_images(dirpath = Directory+'/',BF = name_BF, GFP = name_GFP)

      elif img_set[0].any() and img_set[1].any(): 
        print("Set BW_R: " + str(name))
        imgs = load_images(dirpath = Directory+'/',BF = name_BF, CY5 = name_CY5)

      elif img_set[0].any(): 
        print("Set BW: " + str(name))
        imgs = load_images(dirpath = Directory+'/',BF = name_BF)
        
      invert = True
      imgs_copy = imgs.copy()
      drops = find_drops(imgs, invert)  
      if drops == {}:
        invert = False
        drops = find_drops(imgs_copy, invert)
      if drops != {}: 
        final_data = final_data.append(bead_metrics(drops,Output_Folder), ignore_index=True)
        name_images.append(name)
        if Output_Browser:
          plot_figures(imgs,drops)
      else:
        print("Poor Image Rejected: " + name)
    else:
      print("more images than expected in category: " + name)
  if Output_Folder:
    filename = Output_Folder + "/"+ str(Date) + "_" + Run_ID + "_"

    if Running_On_Mac:
      final_data.to_csv(filename + '.csv')
    else:
      final_data.to_csv("/content/drive/" + filename + '.csv')

  try:
    print("\n\n\n#####################################")
    print("SUMMARY:")
    print("\nMedian")
    print(final_data["Individual Diameter"].median())
    print("\nMode")
    print(final_data["Individual Diameter"].mode())
    print("\n%CV")
    print(variation(final_data["Individual Diameter"]))
    print("\nSummary")
    print(final_data.describe())
    final_data["Individual Diameter"].plot.density()
    plt.title('Size')
  except:
    print("Poor quality images")