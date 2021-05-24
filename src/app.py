import streamlit as st
import warnings
import tkinter as tk
from tkinter import filedialog
import prime.Fluoresence as flu #type: ignore
import prime.Brightfield as bf #type: ignore
import prime.TubeVolume as tube #type: ignore
from PIL import Image
import helper.Tools as tools #type: ignore
import datetime
import pandas as pd

warnings.filterwarnings('ignore')

def run_flu(input_dire):
    cont_list = []
    st.write("Analysis Progress")
    my_bar = st.progress(0)
    directories = tools.get_all_directory(input_dire)
    amount = 1 / len(directories)
    last_amount = 0
    if input_dire:
        for dire in directories:
            cont = flu.Fluorescence_Controller(input_dire, dire)
            my_bar.progress(last_amount + amount)
            last_amount += amount
            cont_list.append(cont)
    return cont_list

def run_bf(input_dire, Output_Image_Browser):
    cont_list = []
    st.write("Analysis Progress")
    my_bar = st.progress(0)
    directories = tools.get_all_directory(input_dire)
    amount = 1 / len(directories)
    last_amount = 0
    if input_dire:
        for dire in directories:
            cont = bf.Brightfield_Controller(dire, Output_Image_Browser)
            my_bar.progress(last_amount + amount)
            last_amount += amount
            cont_list.append(cont)
    return cont_list


def main():
    #INPUTS
    cont_list = []
    Name = st.sidebar.text_input('Name', '')
    Run_ID = st.sidebar.text_input('Run_ID', '')
    Analysis_Type = st.sidebar.selectbox('Type', ('Fluorescence', 'Brightfield', 'Tube Volume Analysis' ))
    CSV_Output_Browser = st.sidebar.checkbox("Output CSV to Browser")
    Image_Output_Browser = st.sidebar.checkbox("Output Images to Browser")
    Date = datetime.date.today()
    # Tkinter Folder Picker #
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    input_dire = []
    clicked = st.sidebar.button('Input Folder Picker')
    if clicked:
        input_dire = st.sidebar.text_input('Selected Input Folder:', filedialog.askdirectory(master=root))
        CSV_Data = []
        if Analysis_Type == "Fluorescence":
            with st.spinner('You are running Fluoresence Analysis...'):
                cont_list = run_flu(input_dire)
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
        if Analysis_Type == "Brightfield":
            with st.spinner('You are running Brightfield Analysis...'):
                output = run_bf(input_dire, Image_Output_Browser)
                if isinstance(output, Exception):
                    st.write(output)
                elif output is not None:
                    for cont in output:
                        with st.spinner("Generating Data..."):
                            CSV_Data = []
                            CSV_Data.append(['Image Name', 
                                            'Image Location', 
                                            'Processed Image Location',
                                            'Analysis Type', 
                                            'Med area', 
                                            'Area STD', 
                                            '#>2x Med', 
                                            'Area>2x Med', 
                                            "Total Area", 
                                            "Emulsion Stability",
                                            '', 
                                            '', 
                                            'Total Drops', 
                                            '%Bead Loading', 
                                            'Number Beads', 
                                            "Number NoBeads"])
                            for data in cont:
                                if Image_Output_Browser:
                                    image = Image.open(data[2])
                                    st.image(image)
                                    st.pyplot(data[3])
                                CSV_Data.append(data)
                            i, filename = tools.Write_CSV(input_dire, Run_ID, CSV_Output_Browser, CSV_Data)
                            if filename is None:
                                st.write(i)
                            if CSV_Output_Browser:
                                df = pd.read_csv(filename + str(i) + ".csv")
                                st.write(df)
                            st.success("Data Generated, Set Complete!")
            st.success("Done!")
            st.balloons()
        if Analysis_Type == "Tube Volume Analysis":
            with st.spinner("Displaying Tube Analysis Images..."):
                fig_list = tube.process_images(path = input_dire + '/*.jpg')
                for fig in fig_list:
                    st.pyplot(fig)
            st.success("Tube Analysis Images Displayed")

if __name__ == '__main__':
    main()