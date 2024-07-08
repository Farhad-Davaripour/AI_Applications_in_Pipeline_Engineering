import streamlit as st
import pandas as pd
from src import tools
from src.tools import plot_anomalies_by_year
import importlib


# Reload the module to reflect the changes
importlib.reload(tools)


# Function to load and cache the DataFrame
@st.cache_data
def load_data():
    return pd.read_csv('Dataset/processed_data/AnomaliesProc_Mapped_All_GirthWelds_Validated.csv')

# Load the data
Anomaly_mapped_df = load_data()

# Get the unique Girth Weld Numbers and calculate min and max values
unique_girth_weld_numbers = Anomaly_mapped_df.GirthWeldNumber.unique()
min_girth_weld_number = unique_girth_weld_numbers.min()
max_girth_weld_number = unique_girth_weld_numbers.max()

# Input field for the Girth Weld Number
GirthWeldNumber = st.number_input("Enter a Girth Weld Number:", min_value=int(min_girth_weld_number), max_value=int(max_girth_weld_number), step=1)

# Plot anomalies by year
plot_anomalies_by_year(Anomaly_mapped_df, GirthWeldNumber, figsize=(12, 3))

st.image(f'artifacts/anomaly_mapping_{GirthWeldNumber}.png')
