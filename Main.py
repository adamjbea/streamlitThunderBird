from Scripts.Tools import get_all_directory
from Scripts import Fluoresence as flu
from datetime import datetime
import streamlit as st
import warnings
import tkinter as tk
from tkinter import filedialog

warnings.filterwarnings('ignore')

#@title RUN / MAIN

# Mostly Error Checking
# Determines what the folder is and what action to use for it
# Collects all data from both types of runs and formats and creates a output
###############################################################################
Name = st.sidebar.text_input('Name', '')
Date = st.sidebar.text_input('Date', datetime.today().strftime('%m/%d/%Y'))
Run_ID = st.sidebar.text_input('Run_ID', '')
Analysis_Type = st.sidebar.selectbox('Type', 
                                    ('Fluoresence', 'placeholder', 'placeholder'))
#input_dire = st.sidebar.text_input('Directory', )
input_files = []
# Set up tkinter
root = tk.Tk()
root.withdraw()

# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)

# Folder picker button
input_dire = []
clicked = st.sidebar.button('Folder Picker')
if clicked:
    input_dire = st.sidebar.text_input('Selected folder:', filedialog.askdirectory(master=root))
directories = get_all_directory(input_dire)
CSV_Data = [[Name,Date,Run_ID,Analysis_Type]]
for dire in directories:
    data = None
    if Analysis_Type == 'Fluoresence':
        CSV_Data.append(['Image Name',"r_intensity_list","g_intensity_list","b_intensity_list","rb_intensity_list","rg_intensity_list","bg_intensity_list"])

    data = flu.Fluorescence_Controller(dire)

    if data is not None:
        for image_data in data:
            CSV_Data.append(image_data)
            CSV_Data.append(data[image_data])

    st.balloons()
#st.write("Folder: " + str(dire) +  "    \n\nFINSIHED\n\n\n")

#write the collected Data
#################################################
#tools.Write_CSV(CSV_Data,custom_name=None)