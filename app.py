import streamlit as st
import warnings
import tkinter as tk
from tkinter import filedialog
import Main as main
from PIL import Image
warnings.filterwarnings('ignore')

#INPUTS
cont_list = []
Name = st.sidebar.text_input('Name', '')
Run_ID = st.sidebar.text_input('Run_ID', '')
Analysis_Type = st.sidebar.selectbox('Type', 
                                    ('Fluoresence', 'placeholder', 'placeholder'))
# Tkinter Folder Picker
root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)
input_dire = []
clicked = st.sidebar.button('Input Folder Picker')
if clicked:
    input_dire = st.sidebar.text_input('Selected Input Folder:', filedialog.askdirectory(master=root))
    input_has_been_clicked = True
    with st.spinner('You are running Fluoresence Analysis...'):
        cont_list = main.run(input_dire)
        if cont_list is not None:
            for cont in cont_list:
                if cont is not None:
                    cont[0].to_csv(input_dire + "/" + "Analyzed" + Name + "_" + Run_ID + "_" + Analysis_Type  + ".csv", index=True)
                    st.write(cont[0])
                    for name, img in cont[1]:
                        st.write("File: ", name)
                        #st.pyplot(img)
                        img.savefig(input_dire + "/" + "Analyzed" + name + ".png")
                        saved_img = Image.open(input_dire + "/" + "Analyzed" + name + ".png")
                        st.image(saved_img)
    st.success("Done!")
    st.balloons()