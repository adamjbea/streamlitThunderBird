from Scripts.Tools import get_all_directory, Write_CSV
from Scripts import Fluoresence as flu
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
clicked = st.sidebar.button('Input Folder Picker')
if clicked:
    input_dire = st.sidebar.text_input('Selected Input Folder:', filedialog.askdirectory(master=root))
    input_has_been_clicked = True
directories = get_all_directory(input_dire)
if input_dire:
    for dire in directories:
        data = None
        new_name = None
        if Analysis_Type == 'Fluoresence':
            cont = flu.Fluorescence_Controller(dire)
        if cont is not None:
            cont[1].to_csv(Name + "_" + Run_ID + "_" + Analysis_Type + "_" + cont[0] + ".csv", index=True)
            st.balloons()
#st.write("Folder: " + str(dire) +  "    \n\nFINSIHED\n\n\n")

#write the collected Data
#################################################
