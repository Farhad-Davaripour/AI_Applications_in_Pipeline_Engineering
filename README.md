# AI Applications in Pipeline Engineering

### Summary
This repository provides a simplified method to map (external corrosion) anomalies across inspection years. Check out the app below to visualize mapped anomalies:

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-applications-in-pipeline-engineering.streamlit.app/)

The main objective is to provide an end-to-end workflow for implementing different machine learning algorithms to:

- **Drive insights**: Use L1 regularization (Lasso algorithm) to identify feature importance.
- **Fill missing values**: Predict the missing max depth of anomalies by training a Regressor model against historical data.
- **Forecast the future max depth of existing anomalies**: Use a modified regressor to predict future anomaly depth.

### Introduction
Pipeline integrity management is crucial in ensuring the safety and reliability of gas and oil transportation. In-line inspection (ILI) tools are extensively used to detect and measure anomalies in pipelines. Accurately predicting the maximum depth of these anomalies is essential for proactive maintenance and risk mitigation. 

This repository demonstrates a comprehensive workflow, from data loading and cleaning to advanced machine learning modeling, aimed at predicting anomaly depths effectively.

### Data Exploration and Cleaning
This involves exploratory data analysis (EDA) to understand the data distribution and identify patterns, handling duplicate records, and managing missing values.

### Feature Engineering
We compute new features such as aspect ratio and area of anomalies, and create cyclic features from angular measurements.

### Anomaly Mapping
We match anomalies across different inspection years to track their growth and changes over time. This involves sophisticated matching algorithms to identify corresponding anomalies based on relative distances and orientations. Use this [link](https://ai-applications-in-pipeline-engineering.streamlit.app/) to visualize the mapped anomalies through different inspection years:

### Modeling
We employ machine learning models, particularly the Hist Gradient Boosting Regressor, to predict the maximum depth of anomalies. This includes data preparation, model training, hyperparameter tuning, and evaluation.

### Prediction and Validation
The predicted values are validated against actual measurements to ensure accuracy. We also compare the machine learning predictions with domain-specific estimates to highlight the added value of advanced modeling techniques.

### Data Source
The ILI data for this study is publicly available from the [Mendeley Data repository](https://data.mendeley.com/datasets/c2h2jf5c54/1). The dataset, titled "Dataset for: Cross-country Pipeline Inspection Data Analysis and Testing of Probabilistic Degradation Models," was published on October 4, 2021, by Rioshar Yarveisy, Faisal Khan, and Rouzbeh Abbassi from Memorial University of Newfoundland and Macquarie University. The dataset includes four consecutive ILI data sets, which lack certain details such as coordinates, likely due to anonymization efforts.
