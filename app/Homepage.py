import streamlit as st

# Homepage content
st.title("AI Applications in Pipeline Engineering")
st.markdown("""
### Summary
This application provides a simplified method to map (external corrosion) anomalies across inspection years. Use the app to visualize mapped anomalies and explore the data.

### Anomaly Mapping
We match anomalies across different inspection years to track their growth and changes over time. This involves sophisticated matching algorithms to identify corresponding anomalies based on relative distances and orientations.

### Data Source
The ILI data for this study is publicly available from the [Mendeley Data repository](https://data.mendeley.com/datasets/c2h2jf5c54/1). The dataset, titled "Dataset for: Cross-country Pipeline Inspection Data Analysis and Testing of Probabilistic Degradation Models," was published on October 4, 2021, by Rioshar Yarveisy, Faisal Khan, and Rouzbeh Abbassi from Memorial University of Newfoundland and Macquarie University. The dataset includes four consecutive ILI data sets, which lack certain details such as coordinates, likely due to anonymization efforts.""")