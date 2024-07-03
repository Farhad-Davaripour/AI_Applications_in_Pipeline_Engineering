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

    def process_all_girth_welds(self):
        # Get all unique girth weld numbers
        girth_weld_numbers = self.df.GirthWeldNumber.unique()

        # Initialize an empty list to store results for all girth weld numbers
        all_results = []

        # Iterate over all girth weld numbers
        for gw_num in girth_weld_numbers:
            print(f"Processing girth weld number: {gw_num}")
            try:
                # Process the data for the current girth weld number
                AnomaliesProc_Mapped_single = self.process_data(girth_weld_number=gw_num)
                
                # Check if the result is empty
                if AnomaliesProc_Mapped_single.empty:
                    print(f"No data found for girth weld number: {gw_num}")
                    continue
                
                # Print the columns of the result for debugging
                print(f"Columns in result: {AnomaliesProc_Mapped_single.columns.tolist()}")
                
                # Check if required columns are present
                required_columns = ['InspectionYear', 'RelativeDistance_m', 'GirthWeldNumber']
                missing_columns = [col for col in required_columns if col not in AnomaliesProc_Mapped_single.columns]
                
                if missing_columns:
                    print(f"Missing columns for girth weld number {gw_num}: {missing_columns}")
                    continue
                
                # Add the results to the list
                all_results.append(AnomaliesProc_Mapped_single)
            except Exception as e:
                print(f"Error processing girth weld number {gw_num}: {str(e)}")

        # Combine all results into a single DataFrame
        if all_results:
            AnomaliesProc_Mapped = pd.concat(all_results, ignore_index=True)

            # Sort the final DataFrame
            AnomaliesProc_Mapped = AnomaliesProc_Mapped.sort_values(['GirthWeldNumber', 'InspectionYear', 'RelativeDistance_m'])

            # Reset the index
            AnomaliesProc_Mapped = AnomaliesProc_Mapped.reset_index(drop=True)

            # Print the shape of the final DataFrame to verify
            print(f"Final AnomaliesProc_Mapped shape: {AnomaliesProc_Mapped.shape}")
            print(f"Columns in final DataFrame: {AnomaliesProc_Mapped.columns.tolist()}")

            return AnomaliesProc_Mapped
        else:
            print("No valid results were processed.")
            return None
    

def plot_anomalies_by_year(anomalies_df, girth_weld_number, figsize=(15, 6)):
    # Filter anomalies for the given girth weld number
    anomalies_df = anomalies_df[anomalies_df.GirthWeldNumber == girth_weld_number]
    
    # Get unique years
    years = sorted(anomalies_df['InspectionYear'].unique())
    num_years = len(years)
    
    # Create subplots
    fig, axes = plt.subplots(num_years, 1, figsize=(figsize[0], figsize[1] * num_years), sharex=True)
    if num_years == 1:
        axes = [axes]
    
    # Find global x limits
    x_min = np.floor(anomalies_df['RelativeDistance_m'].min() * 10) / 10
    x_max = np.ceil(anomalies_df['RelativeDistance_m'].max() * 10) / 10
    
    for i, year in enumerate(years):
        ax = axes[i]
        df_year = anomalies_df[anomalies_df['InspectionYear'] == year]
        
        # Plot anomalies for the year
        new_anomalies = df_year[df_year['Tag'] != 'old']
        old_anomalies = df_year[df_year['Tag'] == 'old']
        
        ax.scatter(new_anomalies['RelativeDistance_m'], new_anomalies['SignificantPointOrientation_deg'], 
                   label='New Anomalies', c='blue', marker='o')
        ax.scatter(old_anomalies['RelativeDistance_m'], old_anomalies['SignificantPointOrientation_deg'], 
                   label='Old Anomalies', c='red', marker='o')
        
        # Annotate anomalies
        for _, row in df_year.iterrows():
            color = 'red' if row['Tag'] == 'old' else 'blue'
            ax.annotate(f'T:{row["Tag"]}', 
                        (row['RelativeDistance_m'], row['SignificantPointOrientation_deg']),
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8, color=color)
        
        # Set y-axis limits and ticks for 45-degree increments
        ax.set_ylim(0, 360)
        ax.set_yticks(np.arange(0, 361, 45))
        
        # Set x-axis limits
        ax.set_xlim(x_min, x_max)
        
        # Customize grid
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='black')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        
        # Set title for each subplot
        ax.set_title(f'Anomalies for Year {year}')
        
        # Set y-axis label
        ax.set_ylabel('Orientation (degrees)')
        
        # Hide x-axis labels for all but the last subplot
        if i < num_years - 1:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        else:
            ax.set_xlabel('Relative Distance (m)')
        
        # Add legend
        ax.legend()
    
    # Set x-axis ticks and labels for the shared x-axis
    fig.axes[-1].set_xticks(np.arange(x_min, x_max + 0.1, 0.1))
    fig.axes[-1].set_xticks(np.arange(x_min, x_max + 0.05, 0.05), minor=True)
    fig.axes[-1].tick_params(axis='x', which='major', labelrotation=45, labelright=True)
    
    plt.tight_layout()
    plt.show()
    
def detect_errors(row, length_change_threshold=10, width_change_threshold=50, depth_change_threshold=0.5):
    """
    Detects if there are errors in the ILI data based on specified thresholds.
    
    Parameters:
    row (pd.Series): A row of data containing ILI measurements.
    length_change_threshold (float): Threshold for significant change in feature length (mm).
    width_change_threshold (float): Threshold for significant change in feature width (mm).
    depth_change_threshold (float): Threshold for significant change in feature depth (mm).
    
    Returns:
    str: "Error" if the row contains an anomaly that exceeds any of the thresholds, otherwise "Okay".
    
    Logic for Thresholds:
    - length_change_threshold = 10 mm: This value is selected because changes in feature length greater than 10 mm are considered significant and may indicate an error or substantial anomaly in the pipeline.
    - width_change_threshold = 50 mm: This threshold is set higher because width measurements can have more variability. A change greater than 50 mm could signal a substantial change in the feature's shape or an error.
    - depth_change_threshold = 0.5 mm: Depth measurements are critical for assessing the severity of anomalies. A change greater than 0.5 mm is significant and could indicate corrosion or another issue requiring attention.
    """
    if abs(row['LengthChange']) > length_change_threshold:
        return "Error"
    if abs(row['WidthChange']) > width_change_threshold:
        return "Error"
    if abs(row['DepthChange']) > depth_change_threshold:
        return "Error"
    return "Okay"

