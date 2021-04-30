from Scripts.Tools import get_all_directory
from Scripts import Fluoresence as flu
import streamlit as st

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
