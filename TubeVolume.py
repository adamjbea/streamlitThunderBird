import numpy as np
import glob
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation, gaussian_filter, median_filter, label, labeled_comprehension
from scipy.signal import medfilt,convolve2d
import matplotlib.image as mpimg 

# Constants
TUBE_PITCH   = 275          # Pixel spacing between tubes
CROP         = [200,200+(TUBE_PITCH+1)*8,730,1270] # Region of interest of image
PIX_MM       = 35         # pixels/mm (Fixed for EVOS objective), used to calculate uL volume
IMAGE_ROTATE = -.25           # Degrees to rotate image
MED_FILT     = 10           # Median filter size for image (pixels)
GAUSS_FILT   = 1            # Gaussian filter size for image (pixels)
GRAY_THRESH  = [.4,.95]     # Set image min and max gray to be [0] and [1] quantile
EDGE_THRESH  = 15.0/255     # Intensity pixel difference defining a tube edge (threshold of derivative)
EMUL_SMOOTH  = 21           # Median filter size for smoothing emulsion derivative (pixels)
WIDTH_SMOOTH = 15           # Sliding average for smoothing tube width (pixels)
SCALE_EMUL   = [.8,-10.4] # Scale integrated emulsion volume by [0] and add [1]
SCALE_OIL    = [1.2,-4]     # Scale integrated oil volume by [0] and add [1]
OFFSETS      = [50,20,450]  # Pixels from top below which emulsion/tube bend,tube bottom might start
TUBE_DIAM_MM = 5.4          # Inner tube diameter in mm (122pxs)

def process_images(path = ''):
    ''' 
    Returns array of tube volumes and also plots tube images with volumes overlayed.
    
    Parameters:
    -----------
    path: string
        Path with directory + filename (w/wildcard) indicating location of image(s) to be analyzed
        
    Returns:
    -----------
    vols: 3D numpy array (float)
        Array of tube volumes in uL - 8 tubes x 2 volumes (emul,oil) x N images 
    '''
    
    # Files to evaluate in directory
    files = np.sort(np.array(glob.glob(path)))
    
    # Set up array of tube volumes for each file, tube, emul/oil
    vols = np.zeros((8,2,len(files)),dtype='float32')
    fig_list = []
    # Iterate through images
    for i in range(len(files)):
        
        # Read and correct images 
        img, img0 = correct_image(mpimg.imread(files[i]))

        # Find lateral tube extents, median tube brightness, and tube widths at all heights
        tube = calc_profiles(img)

        # Find metrics for each tube (level heights and volumes)
        locs,vols[:,:,i] = calc_metrics(tube)

        # Plot images with volumes overlaid   
        fig = plot_volumes(img0,tube,locs,vols[:,:,i],files[i])
        fig_list.append(fig)

    return fig_list

def correct_image(img):
    ''' 
    Rotates, crops, and color-scales an image
    
    Parameters:
    -----------
    img: 3D numpy array
        Image array (x,y,color)
        
    Returns:
    -----------
    img: 2D numpy array
        Grayscale image from 0-1: rotated, cropped, and scaled
    img0: 3D numpy array
        Color image: original, only rotated and cropped
    '''
    
    global IMAGE_ROTATE, CROP, GRAY_THRESH, MED_FILT, GAUSS_FILT

    # Rotate, CROP, filter images
    img  = interpolation.rotate(img,IMAGE_ROTATE) # Rotate image by angle
    img  = img[CROP[2]:CROP[3],CROP[0]:CROP[1],:] # CROP image
    img0 = np.copy(img)                           # Save image original
    img  = np.sum(img,axis=2)                     # Make grayscale
    img  = median_filter(img,size=MED_FILT)       # Median filter
    img  = gaussian_filter(img,sigma=GAUSS_FILT)  # Gaussian filter

    # Normalize image
    mn = np.quantile(img,GRAY_THRESH[0]) 
    mx = np.quantile(img,GRAY_THRESH[1])
    img = (img-mn)/(mx-mn)
    img[img<0] = 0
    img[img>1] = 1
    
    return (img,img0)

def calc_profiles(img):
    ''' 
    Calculates properties of tubes at each y pixel coordinate (profile).
    Includes tube left and right coordinates, tube width, and median image value within tube.
    
    Parameters:
    -----------
    img: 2D numpy array
        Grayscale image, scaled
        
    Returns:
    -----------
    tube: dictionary of 2D numpy arrays
        Contains four keys: left, right, width, and val.
        Each corresponds to a 2D array of values (N vertical pixels x 8 tubes)
            left  - Left extents of tubes (x pixel).
            right - Right extents of tubes (x pixel).
            width - Difference between lefts and rights.
            val   - Median image values between lefts and rights.
        If 'left' or 'right' are missing, return zeros for all values.
     '''    
        
    global EDGE_THRESH, OFFSETS, TUBE_PITCH, EMUL_SMOOTH, WIDTH_SMOOTH
    
    # Initialize tube dictionary
    tube = {}
    ny = img.shape[0]
    tube['left']  = np.zeros((ny,8),dtype=int) # Tubes' left coordinates at each height
    tube['right'] = np.zeros((ny,8),dtype=int) # Tubes' right coordinates at each height
    tube['width'] = np.zeros((ny,8),dtype=int) # Tubes' widths at each height
    tube['val']   = np.zeros((ny,8)) # Tubes' median image values at each height

    # Edge signal is taken from x derivative
    edge = np.diff(img,axis=1)

    # Iterate through 8 tubes
    for i in range(8):

        # Iterate through each y coordinate (pixel)
        for y in range(ny):

            # Take edge signal for individual tube
            ed = edge[y,i*TUBE_PITCH:(i+1)*TUBE_PITCH-1]

            # Left/right extents are first/last instance of edge derivative above threshold
            left_edge = np.where(ed>EDGE_THRESH)[0]
            right_edge = np.where(ed<-EDGE_THRESH)[0]
            
            # If edges found, calculate tube parameters
            if (len(left_edge)>0) and (len(right_edge)>0) and left_edge[0]<right_edge[-1]:
                tube['left'][y,i]  = left_edge[0] + TUBE_PITCH*i # Left coordinate
                tube['right'][y,i] = right_edge[-1] + TUBE_PITCH*i # Right coordinate
                tube['width'][y,i] = tube['right'][y,i] - tube['left'][y,i] # Width
                tube['val'][y,i]   = np.median(img[y,tube['left'][y,i]:tube['right'][y,i]]) # Median value

    # Clean-up and scale median tube value
    tube['val'] = medfilt(np.nan_to_num(tube['val']),[EMUL_SMOOTH,1]) # Smooth emulsion values
    tube['val'] = np.divide(tube['val'],np.max(tube['val'][:OFFSETS[2]+5,:],axis=0)) # Scale to maximum value (0-1)

    # Clean-up and scale tube width
    tube['width'] = convolve2d(tube['width'], np.ones((WIDTH_SMOOTH,1))/WIDTH_SMOOTH) # Smooth widths
    tube['width'] = np.divide(tube['width'],np.max(tube['width'],axis=0)) # Scale to maximum width (0-1)
    
    #remove all nan and inf from our dictionary of numpy arrays so we don't crash
    tube['width'][[np.isnan(tube['width'])]] = 0
    tube['val'][[np.isnan(tube['val'])]] = 0
    tube['width'][[np.isinf(tube['width'])]] = 0
    tube['val'][[np.isinf(tube['val'])]] = 0

    return tube

def calc_metrics(tube):
    ''' 
    Calculates vertical pixel locations (emulsion top/bottom, tube bend/bottom) and tube volumes
    
    Parameters:
    -----------
    tube: dictionary of 2D numpy arrays
        Contains four keys: left, right, width, and value.
        Each corresponds to a 2D array of values (N vertical pixels x 8 tubes)
        [See 'calc_profile']
        
    Returns:
    -----------
    locs: dictionary of 1D arrays
        Contains four keys: Emul_Top, Emul_Bottom, Tube_Bend, Tube_Bot
        Each key corresponds to 1D array of vertical pixel locations for each tube (pixel zero at top)
            Emul_Top  - Location of emulsion top
            Emul_Bot  - Location of emulsion bottom
            Tube_Bend - Location of tube bend
            Tube_Bot  - Location of tube bottom

    vol: 2D numpy array (float)
        Array of tube volumes in uL - 8 tubes x 2 volumes (emul,oil)
    '''    
    global OFFSETS, PIX_MM, SCALE_EMUL, SCALE_OIL, TUBE_DIAM_MM
    
    # Find vertical emulsion locations in tubes
    locs = {}
    
    # y derivative of median tube values
    emul_edge = np.diff(tube['val'][OFFSETS[0]:OFFSETS[2]+5,:],axis=0) 
    
    # Tallest peak is emulsion top
    locs['Emul_Top'] = np.argmax(emul_edge,axis=0)+OFFSETS[0]    
    
    # Lowest valley is emulsion bottom
    locs['Emul_Bot'] = np.argmin(emul_edge,axis=0)+OFFSETS[0]  

    # Tube bend is where width first decreases
    locs['Tube_Bend'] = [np.where(tube['width'][OFFSETS[1]:,i]<.95)[0][0]+OFFSETS[1] for i in range (8)]
                                                             
    # Tube bottom is where width is almost zero
    locs['Tube_Bot'] = [np.where(tube['width'][OFFSETS[2]:,i]<.1)[0][0]+OFFSETS[2]-10 for i in range(8)]

    # Find volumes of liquids by converting tube widths into areas and summing, scaling
    vol = np.zeros((8,2))
    for i in range(8):
        
        # Emulsion volume
        areas = np.power(tube['width'][locs['Emul_Top'][i]:locs['Emul_Bot'][i],i]*TUBE_DIAM_MM/2,2)*np.pi
        vol[i,0] = abs(np.sum(areas)/PIX_MM * SCALE_EMUL[0] + SCALE_EMUL[1])
        
        # Oil volume
        areas = np.power(tube['width'][locs['Emul_Bot'][i]:locs['Tube_Bot'][i],i]*TUBE_DIAM_MM/2,2)*np.pi
        vol[i,1] = abs(np.sum(areas)/PIX_MM * SCALE_OIL[0] + SCALE_OIL[1])

    return (locs,vol)

def plot_volumes(img0,tube,locs,vol,file):
    ''' 
    Plots tube image with emulsion/tube locations and fluid volumes overlaid.
    Also shows total emul/oil volumes in title.
    
    Parameters:
    -----------
    img0: 3D numpy array
        Color image: original, only rotated and cropped
    tube: dictionary of 2D numpy arrays
        Contains four keys: left, right, val, and width.
        Each key corresponds to a 2D array of values (N vertical pixels x 8 tubes)
        [See 'calc_profile']
    locs: dictionary of 1D arrays        
        Contains four keys: Emul_Top, Emul_Bottom, Tube_Bend, Tube_Bot
        Each key corresponds to 1D array of vertical pixel locations for each tube (pixel zero at top)
        [See 'calc_metrics']
    vol: 2D numpy array (float)
        Array of tube volumes in uL - 8 tubes x 2 volumes (emul,oil)
    file: string
        Path/filename of image being analyzed
    '''        
    
    global OFFSETS
    
    fig = plt.figure(figsize=(13,18))
    
    # Plot image
    plt.imshow(img0) 
    plt.xticks([])
    plt.yticks([])
    plt.ylabel(file.split('\\')[-1].split('.')[0],fontsize=16)

    # For each tube
    for i in range(8):

        # Plot top and bottom emulsion level in red
        for j in ['Emul_Top','Emul_Bot']:
            x = [tube['left'][locs[j][i],i]+2,tube['right'][locs[j][i],i]+2]
            plt.plot(x,[locs[j][i],locs[j][i]],c='r')

        # Plot tube bend
        h = locs['Tube_Bend'][i]
        x1 = [tube['left'][h,i]+2,tube['right'][h,i]+2]
        plt.plot(x1,[h,h],'b+',markersize=10)

        # Plot tube bottom
        h = locs['Tube_Bot'][i]
        x2 = [tube['left'][h,i]+2,tube['right'][h,i]+2]

        # Display emulsion volume
        plt.text(np.mean(x),locs['Emul_Top'][i]/2+locs['Emul_Bot'][i]/2,str(int(vol[i,0]))+' uL',
                 ha='center',va='center')

        # Display oil volume
        plt.text(np.mean([x1[0],x1[1],x2[0],x2[1]]),np.mean([locs['Emul_Bot'][i],h]),
                 str(int(vol[i,1]))+' uL',ha='center')

        # Display title
        plt.title('Total Emulsion: %3.0fuL, Total Oil: %3.0fuL'%(np.sum(vol[:,0]),np.sum(vol[:,1])),fontsize=16)

    #plt.show()
    #st.pyplot(fig)

    return fig