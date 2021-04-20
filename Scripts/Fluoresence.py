import matplotlib.image as mpimg
import cv2
import numpy as np
from scipy.ndimage import label, labeled_comprehension
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage.morphology import grey_dilation
import pandas as pd
from .Tools import *
from .Detection import Color_Detection
import streamlit as st
#@title Fluorescence Analysis

SCATTER_THRESH = .4     # Position of threshold between min and max of scatter
MAX_COLOR      = .99    # Place maximum on fluorescence intensity above this quantile
CONVERSION_4X  = 1.54   # Microscope conversion for 4X objective (um/pix)
CONVERSION_10X = 0.625  # Microscope conversion for 10X objective (um/pix)
DROP_BORDER    = 4      # Pixels added to drop border radius to get full drop extent
MIN_DIAM       = 65     # Minimum drop diameter in microns
MIN_DIAM_PERIM = .25    # Minimum diameter to perimeter ratio (~eccentricity). For circle, diam/perim = 1/pi = 0.32
VALID_COLORS   = ['RGB','BF','DAPI','GFP','RFP','CY5'] # Colors that can be loaded from files
VALID_FLUOR    = ['BF','DAPI','GFP','RFP','CY5']       # Fluorescence channels that can be displayed

###############################################################################
def Fluorescence_Controller(dire):
  return_list = []
  #format: [[set1_identity, [[img1,color,name],[img2,color,name],[img3,color,name]]], [set2_identity, [[img1,color,name],[img2,color,name],[img3,color,name]]]]
  identifier_and_images = Flourecence_Image_Reader(dire)
  if identifier_and_images != []:
    with st.spinner('You are running Fluoresence Analysis...'):
      data = Read_Decide_Analyze(identifier_and_images,dire)
    st.success("Done!")
    st.balloons()
    return data

  #return CSV_DATA

###############################################################################
def load_images(dirpath='',**kwargs):
    ''' 
    Loads and normalizes a single set of fluorescent/bright-field image files into variable.
    If 'RBG' argument is given, all data is taken from that image.
    If not, at least 'BF' argument is necessary.
    
    Parameters:
    -----------
    dirpath: string
        Path with directory image(s) to be analyzed
        
    **kwargs: dictionary
        Filename(s) of images to be loaded.
        Arguments should be from set of VALID_COLORS ('RGB','BF','DAPI','GFP'...)
        
    Returns:
    -----------
    imgs: dictionary
        Collection of each fluorescent/BF image as items with keys from VALID_COLORS
        Images are 2D and normalized from 0-1, except for RGB.
        
    '''
    
    global BW_THRESHOLD, MAX_COLOR
    
    imgs = {}

    # Throw error if no BF data available
    if ('BF' not in kwargs.keys()) & ('RGB' not in kwargs.keys()):
        raise NameError('Need BF or RGB argument.')
    
    # Load images from single 'RGB' file
    if 'RGB' in kwargs.keys():
        img = mpimg.imread(dirpath+kwargs['RGB'])*1.0
        imgs['RGB']  = img
        imgs['BF']   = img[:,:,0]
        imgs['GFP']  = img[:,:,1]-img[:,:,0]
        imgs['DAPI'] = img[:,:,2]-img[:,:,0]
        
    # Otherwise load images individually
    else:
        for color in kwargs.keys():    
            if color in VALID_COLORS:
                imgs[color] = mpimg.imread(dirpath+kwargs[color]) 

    # Normalize images. Exclude extreme bright points, and scale to 0-1
    for color in imgs.keys():
        if color in VALID_FLUOR:
            color_img = imgs[color].copy()
            mx = np.quantile( color_img, MAX_COLOR)
            color_img[ color_img > mx ] = mx
            color_img [ color_img < 0 ] = 0
            imgs[color] = color_img/mx
            if len(imgs[color].shape) == 3:
              imgs[color] = Image_Data_Convert(imgs[color], 0, 255, np.uint8)
              imgs[color] = cv2.cvtColor(imgs[color], cv2.COLOR_BGR2GRAY)
    
    mn = np.quantile(imgs['BF'],.05)
    imgs['BF'][imgs['BF']<mn] = mn
    return imgs

###############################################################################
def find_drops(imgs):
    ''' 
    Generates a list of drops and their properties from the 'BF' image.
    
    Parameters:
    -----------
    imgs: dictionary
        Collection of each fluorescent/BF image as items with keys from VALID_COLORS
        
    Returns:
    -----------
    drops: dictionary
        Collection of 1D arrays (each length N, where N is # of drops) associated with drop metrics.
        BF, DAPI, GFP, RFP, CY5 - mean image value inside drop (RFU)
        Diameter  - Diameter in microns
        perimeter - Perimenter in microns
        good      - Boolean, whether the drop is large and round enough
        
    '''
   
    global CONVERSION_4X, MIN_DIAM_PERIM, MIN_DIAM, DROP_BORDER, SCATTER_THRESH
    
    drops={}
    
    # Convert BF to BW image using threshold at the mean of the 10th quanth and 50th quantile
    imgs['BW'] = imgs['BF']>np.mean(np.quantile(imgs['BF'],[.1,.5]))

    # Label drops
    lbl,nlbl = label(imgs['BW'])
    lbls = np.arange(1,nlbl+1)

    # Expand boundaries
    lbl = grey_dilation(lbl, size=(3,3))
    
    # Calculate drop values for each color
    for color in imgs.keys():
        if color in VALID_FLUOR:
            drops[color] = labeled_comprehension(imgs[color], lbl, lbls, np.mean, float, 0)
            if 'RGB' in imgs.keys():
                drops[color] = drops[color] - drops['BF']
                
    # Find drop edges by dilating and keeping borders (to determine perimeter)
    c = convolve2d(imgs['BW'],np.ones((3,3))/9)
    c = np.logical_and(c>.5,c<1)
    lbl_borders = np.multiply(c[1:-1,1:-1],lbl)
                
    # Find area and perimeter
    pixel_area         = labeled_comprehension(imgs['BW']>-1, lbl, lbls, np.sum, float, 0)
    drops['Radius']  = np.sqrt(pixel_area/np.pi)
    drops['Diameter']  = (np.sqrt(pixel_area/np.pi)*2+DROP_BORDER) * CONVERSION_4X
    drops['perimeter'] = labeled_comprehension(imgs['BW']>-1, lbl_borders , lbls, np.sum, float, 0) * CONVERSION_4X

    mx,my = np.meshgrid(np.arange(imgs['BW'].shape[1]),np.arange(imgs['BW'].shape[0]))
    drops['X']         = labeled_comprehension(mx, lbl , lbls, np.mean, float, 0)
    drops['Y']         = labeled_comprehension(my, lbl , lbls, np.mean, float, 0)

    # Exclude regions with small area-perimeter ratios (high eccentricity) or small areas
    good = (np.divide(drops['Diameter'],drops['perimeter']+20)>MIN_DIAM_PERIM) & (drops['Diameter']>MIN_DIAM)
    drops['good'] = good
    
    # Generate a BW mask of only good drops
    lbl = lbl.flatten()
    lbl = np.isin(lbl,np.where(drops['good'])[0]+1)
    imgs['BW'] = lbl.reshape(imgs['BW'].shape[0],imgs['BW'].shape[1])      


    # Generate histogram data and thresholds
    drops['hists'] = []
    drops['bins'] = []


    colors = np.sort(list(drops.keys()))
    colors = np.insert(colors[colors!='Diameter'],0,'Diameter')
    colors = [c for c in colors if c in ['DAPI','GFP','RFP','CY5','Diameter']]
    nc = len([c for c in colors if c in ['DAPI','GFP','RFP','CY5','Diameter']])

    drops['colors'] = colors
    drops['color_thresh'] = np.zeros(nc)

    for i in range(nc):
        h,b = np.histogram(drops[colors[i]][good],60)
        h = np.insert(h,[0,len(h)],[0,0])                 # Pad with zeros
        b = np.insert(b,[0,len(b)-1],[2*b[0]-b[1],b[-1]]) # Pad with zeros
        h = np.sqrt(h) # Sqrt better represents small values
        drops['hists'].append(h)
        drops['bins'].append(b)

        b = drops['bins'][i][:-1]
        di = drops[colors[i]][good]

        # Default threshold is 40% from bottom to top
        drops['color_thresh'][i] = SCATTER_THRESH*np.max(di)+(1-SCATTER_THRESH)*np.min(di)

        # If a lower cluster is found, set thresh at FHWM + 2 bins
        mnpk = np.argmax(np.multiply(h,b<drops['color_thresh'][i]))
        fwhm = np.where((b>b[mnpk]) & (h<h[mnpk]/2))[0]
        if (fwhm.size>0) and (fwhm[0]-mnpk<10):
           drops['color_thresh'][i] = b[fwhm[0]+2]
    
    
    return drops

###############################################################################
def plot_figures(imgs,drops):
    ''' 
    Plots masked fluorescent images (where there are drops) and 2D scatters of Diameter/fluorescent combinations.
    
    Parameters:
    -----------
    imgs: dictionary
        Collection of each fluorescent/BF image as items with keys from VALID_COLORS
        

    drops: dictionary
        Collection of 1D arrays (each length N, where N is # of drops) associated with drop metrics.
        BF, DAPI, GFP, RFP, CY5 - mean image value inside drop (RFU)
        Diameter  - Diameter in microns
        perimeter - Perimenter in microns
        good      - Boolean, whether the drop is large and round enough
        Radius    - Diameter in pixels
        X         - X Center, in pixels
        Y         - Y Center, in pixels

        1D array of with one element per color
        color_thresh - Fluorescent/diameter threshold per color
    '''
    global MIN_DIAM, MIN_DIAM_PERIM

    # Organize channels
    colors = drops['colors']
    nc = len(colors) # Number of colors
    nx  = nc*(nc-1)+1 # Number of color combinations

    # Screen colors associated with channels (0=Red, 1=Green, 2=Blue, -1= BW)
    chancol = {'CY5':0, 'RFP':0, 'GFP':1,'DAPI':2,'Diameter':-1}
    
    good = drops['good']

    ### Plot BF/fluorescent images
    fig = plt.figure(figsize=(nx*8*1.5,15*1.5))
    for i in range(nc):
        plt.subplot(2,nc+1,i+1)
        # Generate color-scheme for image
        vals = np.ones((256, 4))
        for j in range(3):
            vals[:, j] = np.linspace(0, chancol[colors[i]] in [-1,j], 256)
            
        # Plot each image
        if colors[i]=='Diameter':
            plt.imshow(imgs['BF'],cmap=ListedColormap(vals))
            drops_show = good
            outline_color = 'r'
        else:
            plt.imshow(np.multiply(imgs[colors[i]],imgs['BW']),cmap=ListedColormap(vals)) # Mask fluorescence by BW
            di = drops[colors[i]]
            drops_show = good & (di>drops['color_thresh'][i])
            outline_color = np.isin([0,1,2],chancol[colors[i]],invert=True)
        # Draw drop outlines on image
        th = np.arange(0,2*3.141593,3.141593/20)
        _,XX = np.meshgrid(th,drops['X'][drops_show])
        _,YY = np.meshgrid(th,drops['Y'][drops_show])
        TH,RR = np.meshgrid(th,drops['Radius'][drops_show]+10)
        if isinstance(outline_color, str):
          plt.scatter(x=XX+np.multiply(np.cos(TH),RR),y=YY+np.multiply(np.sin(TH),RR),color=outline_color,s=.5)
        else:
          plt.scatter(x=XX+np.multiply(np.cos(TH),RR),y=YY+np.multiply(np.sin(TH),RR),c=outline_color,s=.5)

        plt.xticks([])
        plt.yticks([])
        plt.xlim([0, imgs['BW'].shape[1]])
        plt.ylim([0, imgs['BW'].shape[0]])
        plt.title(colors[i] + ' (Masked)' if colors[i]!='Diameter' else 'BF',fontsize=20)
    # Plot merged image if it exists

    if 'RGB' in imgs.keys():
        st.write("rgb in imgs.keys")
        plt.subplot(1,nc+1,nc+1)
        plt.imshow(np.multiply(imgs['RGB'],imgs['BW'][:,:,None]))
        plt.xticks([])
        plt.yticks([])
        plt.title('RGB (Masked)',fontsize=20)

    ### Plot scatters combinations
    cnt = 1 # Combination iteration count
    # Debugging figure. See what drops it finds
    plt.subplot(2,nx,nx+cnt)
    plt.loglog(drops['Diameter']+.05,np.divide(drops['Diameter'],drops['perimeter']+20),'.',markersize=1)
    plt.plot(plt.xlim(),[MIN_DIAM_PERIM,MIN_DIAM_PERIM],'k--')
    plt.plot([MIN_DIAM,MIN_DIAM],plt.ylim(),'k--')
    plt.text(plt.xlim()[0]*1.5,(MIN_DIAM_PERIM+plt.ylim()[0])/2,'Small, Not Round\n(BAD)',color=[1,0,0],va='center',ha='left')
    plt.text(plt.xlim()[1]/1.5,(MIN_DIAM_PERIM+plt.ylim()[0])/2,'Large, Not Round\n(BAD)',color=[1,0,0],va='center',ha='right')
    plt.text(plt.xlim()[0]*1.5,(MIN_DIAM_PERIM+plt.ylim()[1])/2,'Small, Round\n(BAD)',color=[1,0,0],va='center',ha='left')
    plt.text(plt.xlim()[1]/1.5,(MIN_DIAM_PERIM+plt.ylim()[1])/2,'Large, Round\n(GOOD)',color=[0,.5,0],va='center',ha='right')
    plt.xlabel('Diameter (um)')
    plt.ylabel('Roundness (Diameter/Perimeter)')
    plt.title('Roundness vs Diameter')
    cnt +=1
    # Iterate through each pair of BF/fluorescences
    for i in range(nc):
        for j in range(i+1,nc):
            
            # Collect coordinates
            di = drops[colors[i]][good]
            dj = drops[colors[j]][good]
            
            # Find quadrant thresholds
            yth = drops['color_thresh'][j] if colors[j]!='Diameter' else 76
            xth = drops['color_thresh'][i] if colors[i]!='Diameter' else 76 # Diameter threshold is 76um
            
            # Scatter colors
            col = np.ones((4,3))*.2
            for k in range(2):
                ind = [i,j][k] if colors[i]=='Diameter' else [j,i][k]
                if chancol[colors[ind]]>=0:
                    col[[k+1,3],chancol[colors[ind]]] = [.65,.9]
                    
            # Plot scatters
            plt.subplot(2,nx,nx+cnt)
            plt.scatter(di,dj,s=2,c=col[(dj>yth)+2*(di>xth),:])
            plt.xlabel(colors[i]+' (RFU)' if colors[i]!='Diameter' else 'Diameter (um)',fontsize=14)
            plt.ylabel(colors[j]+' (RFU)' if colors[j]!='Diameter' else 'Diameter (um)',fontsize=14)
            plt.title(colors[j]+' vs '+colors[i],fontsize=16)
            
            # Plot four-quadrant (or upper/lower) counts on scatters
            for m in range(2):
                mm = dj*(m-.5)<=yth*(m-.5)
                if colors[i]!='Diameter':
                    for k in range(2):
                        kk = di*(k-.5)<=xth*(k-.5)
                        if np.sum(kk&mm)>0:
                            plt.text(np.quantile(di[kk & mm],.8),np.quantile(dj[kk & mm],.8),str(np.sum([kk&mm])),fontsize=16)
                else:
                    if np.sum(mm)>0:
                        plt.text(np.quantile(di,.8),np.quantile(dj[mm],.8),str(np.sum([mm])),fontsize=16)


            
            # Plot x histogram
            bins = drops['bins']
            hists = drops['hists']
            xlim = plt.xlim()
            plt.plot((hists[j]/np.max(hists[j]))*np.diff(xlim)*.15+xlim[0],bins[j][:-1],'k')
            plt.xlim(xlim)

            # Plot y histogram
            ylim = plt.ylim()
            ylim = [ylim[0]-np.diff(ylim)*.2,ylim[1]]
            plt.plot(bins[i][:-1],hists[i]/np.max(hists[i])*np.diff(ylim)*.15+ylim[0],'k')
            plt.ylim(ylim)
            plt.plot(plt.xlim(),[yth,yth],'k--')
            if colors[i]!='Diameter':
                plt.text(np.mean(plt.xlim()),plt.ylim()[1],'Above (%3.1f%%), Right (%3.1f%%), Above+Right (%3.1f%%)'%
                          (100.0*np.sum(dj>yth)/len(dj),
                          100.0*np.sum(di>xth)/len(dj),
                          100.0*np.sum((di>xth) & (dj>yth))/len(dj)),va='top',ha='center',fontsize=12)
                
                plt.plot([xth,xth],plt.ylim(),'k--')
            else:
                plt.text(np.mean(plt.xlim()),plt.ylim()[1],'Above (%3.1f%%)'%
                          (100.0*np.sum(dj>yth)/len(dj)),va='top',ha='center',fontsize=12)

            cnt += 1
    #plt.show()
    st.pyplot(fig)
    

###############################################################################
def fluor_metrics(drops):

    ''' 
    Return DataFrame 'metrics' with drop readouts
    
    Parameters:
    -----------
    drops: dictionary
        Collection of 1D arrays (each length N, where N is # of drops) associated with drop metrics.
        BF, DAPI, GFP, RFP, CY5 - mean image value inside drop (RFU)
        Diameter  - Diameter in microns
        perimeter - Perimenter in microns
        good      - Boolean, whether the drop is large and round enough

    Returns:
    -----------
    metrics: DataFrame
        Single index, multicolumn DataFrame, containing metrics for image set
        Fraction of drops above fluorescence threshold for each channel
        Median drop diameter
        Area of drop above 
    '''

    global SCATTER_THRESH

    # Organize channels
    colors = np.sort(list(drops.keys()))
    colors = np.insert(colors[colors!='Diameter'],0,'Diameter')
    colors = [c for c in colors if c in ['DAPI','GFP','RFP','CY5','Diameter']]
    nc = len(colors)

    metrics = pd.DataFrame()
    good = drops['good']

    # Iterate through channels
    for i in range(nc):
        d = drops[colors[i]][good]
        metrics[colors[i]+'+'] = [np.sum((d>drops['color_thresh'][i]))/len(d)]
        metrics[colors[i]+'+ Count'] = [np.sum((d>drops['color_thresh'][i]))]

    # Calculate non-fluorescent metrics
    drop_count = np.sum(good)
    metrics['Drop_Count'] = drop_count

    diam = drops['Diameter'][good]
    med = np.median(diam)
    metrics['MedDropDiam'] = med
    metrics['Area>2XMed'] = np.sum(np.square(diam[np.square(diam)>2*med**2]))/np.sum(np.square(diam))

    return(metrics)
    

###############################################################################
def Flourecence_Image_Reader(dire):
  uploaded = []
  return_list = []
  uploaded,filenames = all_imgs_directory(dire)
  st.write("Uploaded: ", uploaded)
  st.write("Directory: ", dire)
  filenames = [filename.replace(dire + '/', '') for filename in filenames]
  
  #format: [[set_identity, [[img1,color,name],[img2,color,name],[img3,color,name]]], [img_identity, [[img1,color,name],[img2,color,name],[img3,color,name]]]]
  identifier_and_images = []
  
  if filenames:
    st.write("Filenames: ", filenames)

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
      st.write("pre image name ", filenames[image_iter_folder])
      j = 0
      image_name = filenames[image_iter_folder].replace(dire, '')
      st.write("Image_name: ", image_name)
      reversed_name = image_name[::-1]
      st.write("reversed name ", reversed_name)
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


######################################################################################
def Read_Decide_Analyze(identifier_and_images,Directory, Output_Browser=True):

  st.write("IDIMGSET: ", identifier_and_images)
  final_data = pd.DataFrame()
  name_images = []
 # what type of image set is this? Then run analysis
  for image_set in identifier_and_images:
    print("\n\n#############################")
    #reverse back for human readability
    name = image_set[0][::-1]
    st.write("Name: ,", name)
    number_pictures = len(image_set[1])
    name_Mixed = None
    if number_pictures <= 4:
      empty_array = np.zeros((1,1))
      img_set = [empty_array,empty_array,empty_array,empty_array]
      
      for i in range(len(image_set[1])):
        st.write("I", i)
        color = image_set[1][i][1]
        st.write("Color: ", color)
        image = image_set[1][i][0]
        
        if color == "grey":
          name_BF = image_set[1][i][2]
          img_set[0] = image
          st.write(image_set[1][i][2])

        elif color == "red":
          name_CY5 = image_set[1][i][2]
          img_set[1] = image
          st.write(image_set[1][i][2])

        elif color == "blue": 
          name_DAPI = image_set[1][i][2]
          img_set[2] = image
          st.write(image_set[1][i][2])

        elif color == "green":
          name_GFP = image_set[1][i][2]
          img_set[3] = image
          st.write(image_set[1][i][2])

        elif color == "bf_mixed":
          name_Mixed = image_set[1][i][2]
          st.write(image_set[1][i][2])

      st.write("0: ", img_set[0].any())
      st.write("1: ", img_set[1].any())
      st.write("2: ", img_set[2].any())
      st.write("3: ", img_set[3].any())


      if img_set[0].any() and img_set[2].any() and img_set[3].any():    
        st.write("Set BW_B_G: ", name)
        imgs = load_images(dirpath = Directory+'/',BF = name_BF, GFP = name_GFP, DAPI = name_DAPI)
      elif img_set[0].any() and img_set[1].any() and img_set[3].any(): 
        st.write("Set BW_R_G: ", name)
        imgs = load_images(dirpath = Directory+'/',BF = name_BF, GFP = name_GFP, CY5 = name_CY5)
      elif img_set[0].any() and img_set[1].any(): 
        st.write("Set BW_R: ", name)
        imgs = load_images(dirpath = Directory+'/',BF = name_BF, CY5 = name_CY5)
      elif img_set[0].any() and img_set[3].any(): 
        st.write("Set BW_G: ", name)
        imgs = load_images(dirpath = Directory+'/',BF = name_BF, GFP = name_GFP)
      elif img_set[0].any() and img_set[2].any(): 
        st.write("Set BW_B: ", name)
        imgs = load_images(dirpath = Directory+'/',BF = name_BF, DAPI = name_DAPI)
      elif name_Mixed is not None:
        st.write("Set RGB: ", name)
        imgs = load_images(dirpath = Directory+'/',RGB = name_Mixed)
      else:
        st.write("No Process Match: ", name)
      
      try:
        drops = find_drops(imgs)
        final_data = final_data.append(fluor_metrics(drops), ignore_index=True)
        st.write(final_data)
        name_images.append(name)
        plot_figures(imgs,drops)
      except Exception as e:
        st.write("Error:", e)
    
    else:
      print("more images than expected in category: " + name)
  return final_data