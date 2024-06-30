import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Anomaly_mapping:
    def __init__(self, df):
        self.df = df

    def sort_dataframe(self):
        return self.df.sort_values(['GirthWeldNumber', 'InspectionYear', 'RelativeDistance_m'])

    def filter_girthweld(self, girth_weld_number):
        return self.df[self.df.GirthWeldNumber == girth_weld_number]

    def get_unique_years(self):
        return self.df['InspectionYear'].unique()

    def process_years(self, years):
        results = []
        for i in range(1, len(years)):
            results.extend(self.match_anomalies(years[i-1], years[i]))
        return results

    def match_anomalies(self, previous_year, current_year):
        previous_year_data = self.df[self.df['InspectionYear'] == previous_year].reset_index(drop=True)
        current_year_data = self.df[self.df['InspectionYear'] == current_year].copy().reset_index(drop=True)
        matches = []

        for _, anomaly in current_year_data.iterrows():
            previous_year_data['DistanceDiff'] = np.abs(anomaly['RelativeDistance_m'] - previous_year_data['RelativeDistance_m'])
            previous_year_data['OrientationDiff'] = np.abs(anomaly['SignificantPointOrientation_deg'] - previous_year_data['SignificantPointOrientation_deg'])

            potential_matches = previous_year_data[(previous_year_data['DistanceDiff'] <= 0.05) & 
                                                (previous_year_data['OrientationDiff'] <= 5)].copy()

            if not potential_matches.empty:
                potential_matches['TotalDiff'] = potential_matches['DistanceDiff'] + potential_matches['OrientationDiff']
                closest_match_index = potential_matches['TotalDiff'].idxmin()
                closest_match = previous_year_data.loc[closest_match_index]
                matches.append(self.create_result_dict(anomaly, closest_match, previous_year, True))
            else:
                matches.append(self.create_result_dict(anomaly, None, previous_year, False))

        return matches

    def process_first_year(self, first_year):
        first_year_data = self.df[self.df['InspectionYear'] == first_year].reset_index(drop=True)
        matches = []

        for _, anomaly in first_year_data.iterrows():
            matches.append(self.create_result_dict(anomaly, None, first_year, False))

        return matches

    def create_result_dict(self, anomaly, closest_match, previous_year, is_old):
        if is_old:
            length_change = anomaly['FeatureLength_mm'] - closest_match['FeatureLength_mm']
            width_change = anomaly['FeatureWidth_mm'] - closest_match['FeatureWidth_mm']
            depth_change = anomaly['MaxDepth_mm'] - closest_match['MaxDepth_mm']
            distance_diff = anomaly['RelativeDistance_m'] - closest_match['RelativeDistance_m']
            orientation_diff = anomaly['SignificantPointOrientation_deg'] - closest_match['SignificantPointOrientation_deg']
            return {
                'GirthWeldNumber': anomaly['GirthWeldNumber'],
                'InspectionYear': anomaly['InspectionYear'],
                'RelativeDistance_m': anomaly['RelativeDistance_m'],
                'FeatureLength_mm': anomaly['FeatureLength_mm'],
                'FeatureWidth_mm': anomaly['FeatureWidth_mm'],
                'MaxDepth_mm': anomaly['MaxDepth_mm'],
                'SignificantPointOrientation_deg': anomaly['SignificantPointOrientation_deg'],
                'Prev_InspectionYear': previous_year,
                'Prev_RelativeDistance_m': closest_match['RelativeDistance_m'],
                'Prev_FeatureLength_mm': closest_match['FeatureLength_mm'],
                'Prev_FeatureWidth_mm': closest_match['FeatureWidth_mm'],
                'Prev_MaxDepth_mm': closest_match['MaxDepth_mm'],
                'Prev_SignificantPointOrientation_deg': closest_match['SignificantPointOrientation_deg'],
                'LengthChange': length_change,
                'WidthChange': width_change,
                'DepthChange': depth_change,
                'DistanceDiff': distance_diff,
                'OrientationDiff': orientation_diff,
                'Tag': 'old'
            }
        else:
            return {
                'GirthWeldNumber': anomaly['GirthWeldNumber'],
                'InspectionYear': anomaly['InspectionYear'],
                'RelativeDistance_m': anomaly['RelativeDistance_m'],
                'FeatureLength_mm': anomaly['FeatureLength_mm'],
                'FeatureWidth_mm': anomaly['FeatureWidth_mm'],
                'MaxDepth_mm': anomaly['MaxDepth_mm'],
                'SignificantPointOrientation_deg': anomaly['SignificantPointOrientation_deg'],
                'Prev_InspectionYear': np.nan,
                'Prev_RelativeDistance_m': np.nan,
                'Prev_FeatureLength_mm': np.nan,
                'Prev_FeatureWidth_mm': np.nan,
                'Prev_MaxDepth_mm': np.nan,
                'Prev_SignificantPointOrientation_deg': np.nan,
                'LengthChange': np.nan,
                'WidthChange': np.nan,
                'DepthChange': np.nan,
                'DistanceDiff': np.nan,
                'OrientationDiff': np.nan,
                'Tag': 'new'
            }

    def process_data(self, girth_weld_number):
        self.df = self.sort_dataframe()
        self.df = self.filter_girthweld(girth_weld_number)
        years = self.get_unique_years()

        results = []
        if len(years) >= 1:
            results.extend(self.process_first_year(years[0]))
        if len(years) > 1:
            results.extend(self.process_years(years))

        anomalies_changes = pd.DataFrame(results)
        anomalies_changes = anomalies_changes.sort_values(['InspectionYear', 'RelativeDistance_m'])

        self.df = anomalies_changes.merge(
            self.df, 
            on=['GirthWeldNumber', 'RelativeDistance_m', 'InspectionYear', 'SignificantPointOrientation_deg'], 
            how='left', 
            suffixes=('', '_dup')
        )
        self.df = self.df.drop(columns=[col for col in self.df.columns if '_dup' in col])

        return self.df
    

def plot_anomalies(anomalies_df, year1, year2):
    # Filter data for the two consecutive years
    df_year1 = anomalies_df[anomalies_df['InspectionYear'] == year1]
    df_year2 = anomalies_df[anomalies_df['InspectionYear'] == year2]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15,12))
    
    # Plot anomalies for year1
    ax.scatter(df_year1['RelativeDistance_m'], df_year1['SignificantPointOrientation_deg'], label=f'Year {year1}', c='blue', marker='o')
    
    # Annotate anomalies for year1
    for i, row in df_year1.iterrows():
        ax.annotate(f'T:{row["Tag"]}', 
                    (row['RelativeDistance_m'], row['SignificantPointOrientation_deg']),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='blue')
    
    # Plot anomalies for year2
    ax.scatter(df_year2['RelativeDistance_m'], df_year2['SignificantPointOrientation_deg'], label=f'Year {year2}', c='red', marker='x')
    
    # Annotate anomalies for year2
    for i, row in df_year2.iterrows():
        ax.annotate(f'T:{row["Tag"]}', 
                    (row['RelativeDistance_m'], row['SignificantPointOrientation_deg']),
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color='red')
    
    # Set plot labels and title
    ax.set_xlabel('Relative Distance (m)')
    ax.set_ylabel('Orientation (degrees)')
    ax.set_title(f'Anomalies Comparison for Year {year1} and {year2}')
    ax.legend()
    
    # Set y-axis limits and ticks for 10-degree increments
    ax.set_ylim(0, 360)
    ax.set_yticks(np.arange(0, 361, 10))
    
    # Add minor y-ticks for 5-degree increments
    ax.set_yticks(np.arange(0, 361, 5), minor=True)
    
    # Set x-axis limits and ticks for 10 cm (0.1 m) increments
    x_min = np.floor(min(df_year1['RelativeDistance_m'].min(), df_year2['RelativeDistance_m'].min()) * 10) / 10
    x_max = np.ceil(max(df_year1['RelativeDistance_m'].max(), df_year2['RelativeDistance_m'].max()) * 10) / 10
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(np.arange(x_min, x_max + 0.1, 0.1))
    
    # Add minor x-ticks for 5 cm increments
    ax.set_xticks(np.arange(x_min, x_max + 0.05, 0.05), minor=True)
    
    # Customize grid
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()
    