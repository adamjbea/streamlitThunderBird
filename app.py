import streamlit as st
import warnings
import tkinter as tk
from tkinter import filedialog
import glob
import Fluoresence as flu

warnings.filterwarnings('ignore')

def get_all_directory(input_dir):
    directories = []
    if input_dir:
        if input_dir[-1] != '/':
            input_dir += '/'
        directories = glob.glob(input_dir + "**/", recursive=True)
    return directories

def run(input_dire):
    cont_list = []
    st.write("Analysis Progress")
    my_bar = st.progress(0)
    directories = get_all_directory(input_dire)
    amount = 1 / len(directories)
    last_amount = 0
    if input_dire:
        for dire in directories:
            cont = flu.Fluorescence_Controller(input_dire, dire)
            my_bar.progress(last_amount + amount)
            last_amount += amount
            cont_list.append(cont)
    return cont_list

def main():
    #INPUTS
    cont_list = []
    Name = st.sidebar.text_input('Name', '')
    Run_ID = st.sidebar.text_input('Run_ID', '')
    Analysis_Type = st.sidebar.selectbox('Type', ('Fluorescence', 'Brightfield', ))
    CSV_Output_Browser = st.sidebar.checkbox("Output CSV to Browser")
    Image_Output_Browser = st.sidebar.checkbox("Output Images to Browser")
    # Tkinter Folder Picker #
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    input_dire = []
    clicked = st.sidebar.button('Input Folder Picker')
    if clicked:
        input_dire = st.sidebar.text_input('Selected Input Folder:', filedialog.askdirectory(master=root))
        if Analysis_Type == "Fluorescence":
            with st.spinner('You are running Fluoresence Analysis...'):
                cont_list = run(input_dire)
                if cont_list is not None:
                    for cont in cont_list:
                        if cont is not None:
                            if CSV_Output_Browser:
                                cont[0].to_csv(input_dire + "/" + "Analyzed" + Name + "_" + Run_ID + "_" + Analysis_Type  + ".csv", index=True)
                                st.write(cont[0])
                            with st.spinner("Displaying images for the run..."):
                                for name, img in cont[1]:
                                    if Image_Output_Browser:
                                        st.write("File: ", name)
                                        st.pyplot(img)
                                    img.savefig(input_dire + "/" + "Analyzed" + name + ".png")
                            st.success("Images Displayed! Set Complete")
            st.success("Done!")
            st.balloons()

if __name__ == '__main__':
    main()