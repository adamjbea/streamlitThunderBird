from Scripts.Tools import get_all_directory
from Scripts import Fluoresence as flu
import streamlit as st

def run(input_dire):
    cont_list = []
    directories = get_all_directory(input_dire)
    if input_dire:
        for dire in directories:
            cont = flu.Fluorescence_Controller(input_dire, dire)
            cont_list.append(cont)
    return cont_list
