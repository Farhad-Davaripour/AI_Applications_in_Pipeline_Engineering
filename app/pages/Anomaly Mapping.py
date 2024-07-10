import os
import streamlit as st
import pandas as pd
import importlib
import sys

update_parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..'))
sys.path.append(update_parent_directory)

from src import tools
from src.tools import plot_anomalies_by_year

st. set_page_config(layout="wide")

st.header("Anomaly Mapping")

# Reload the module to reflect the changes
importlib.reload(tools)

save_path = (f'Dataset/processed_data/Plot_Mapped_Anomalies.csv')

# Function to load and cache the DataFrame
@st.cache_data
def load_data():
    return pd.read_csv(save_path)

# Load the data
Anomaly_mapped_df = load_data()

# Get the unique Girth Weld Numbers and calculate min and max values
unique_girth_weld_numbers = Anomaly_mapped_df.GirthWeldNumber.unique()
min_girth_weld_number = unique_girth_weld_numbers.min()
max_girth_weld_number = unique_girth_weld_numbers.max()

# Input field for the Girth Weld Number
GirthWeldNumber = st.number_input("Enter a joint number:", min_value=int(min_girth_weld_number), max_value=int(max_girth_weld_number), step=1)

# Plot anomalies by year
plot_anomalies_by_year(Anomaly_mapped_df, GirthWeldNumber, figsize=(12, 3))

st.image(f'artifacts/anomaly_mapping_{GirthWeldNumber}.png')
