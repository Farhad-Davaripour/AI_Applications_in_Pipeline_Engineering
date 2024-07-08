import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

class Anomaly_mapping:
    def __init__(self, df, relative_distance_threshold=0.1, orientation_threshold=10):
        self.df = df
        self.relative_distance_threshold = relative_distance_threshold
        self.orientation_threshold = orientation_threshold

    def sort_dataframe(self, df):
        return df.sort_values(['GirthWeldNumber', 'InspectionYear', 'RelativeDistance_m'])

    def filter_girthweld(self, df, girth_weld_number):
        return df[df.GirthWeldNumber == girth_weld_number]

    def get_unique_years(self, df):
        return df['InspectionYear'].unique()

    def process_years(self, df, years):
        results = []
        for i in range(1, len(years)):
            results.extend(self.match_anomalies(df, years[i-1], years[i]))
        return results

    def match_anomalies(self, df, previous_year, current_year):
        previous_year_data = df[df['InspectionYear'] == previous_year].reset_index(drop=True)
        current_year_data = df[df['InspectionYear'] == current_year].copy().reset_index(drop=True)
        matches = []

        for _, anomaly in current_year_data.iterrows():
            previous_year_data['DistanceDiff'] = np.abs(anomaly['RelativeDistance_m'] - previous_year_data['RelativeDistance_m'])
            previous_year_data['OrientationDiff'] = np.abs(anomaly['SignificantPointOrientation_deg'] - previous_year_data['SignificantPointOrientation_deg'])

            potential_matches = previous_year_data[(previous_year_data['DistanceDiff'] <= self.relative_distance_threshold) & 
                                                (previous_year_data['OrientationDiff'] <= self.orientation_threshold)].copy()

            if not potential_matches.empty:
                potential_matches['TotalDiff'] = potential_matches['DistanceDiff'] + potential_matches['OrientationDiff']
                closest_match_index = potential_matches['TotalDiff'].idxmin()
                closest_match = previous_year_data.loc[closest_match_index]
                matches.append(self.create_result_dict(anomaly, closest_match, previous_year, True))
            else:
                matches.append(self.create_result_dict(anomaly, None, previous_year, False))

        return matches

    def process_first_year(self, df, first_year):
        first_year_data = df[df['InspectionYear'] == first_year].reset_index(drop=True)
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
        df_sorted = self.sort_dataframe(self.df)
        df_filtered = self.filter_girthweld(df_sorted, girth_weld_number)
        years = self.get_unique_years(df_filtered)

        results = []
        if len(years) >= 1:
            results.extend(self.process_first_year(df_filtered, years[0]))
        if len(years) > 1:
            results.extend(self.process_years(df_filtered, years))

        anomalies_changes = pd.DataFrame(results)
        anomalies_changes = anomalies_changes.sort_values(['InspectionYear', 'RelativeDistance_m'])

        df_merged = anomalies_changes.merge(
            self.df, 
            on=['GirthWeldNumber', 'RelativeDistance_m', 'InspectionYear', 'SignificantPointOrientation_deg'], 
            how='left', 
            suffixes=('', '_dup')
        )
        df_final = df_merged.drop(columns=[col for col in df_merged.columns if '_dup' in col])

        return df_final

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
    # Ensure the 'artifacts' directory exists
    os.makedirs('artifacts', exist_ok=True)
    
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
    fig.axes[-1].set_xticks(np.arange(x_min, x_max + 0.1, 0.5))  # Set major ticks every 50 cm
    fig.axes[-1].set_xticks(np.arange(x_min, x_max + 0.05, 0.05), minor=True)  # Optional: Set minor ticks every 5 cm
    fig.axes[-1].tick_params(axis='x', which='major', labelrotation=45, labelright=True)
    
    plt.tight_layout(pad=2.0)
    plt.savefig(f'artifacts/anomaly_mapping_{girth_weld_number}.png')
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

def rename_anomaly_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the columns of the Anomalies DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame with original column names.

    Returns:
    pd.DataFrame: The DataFrame with renamed columns.
    """
    # Dictionary mapping old column names to new column names
    column_mapping = {
        'Year': 'InspectionYear',
        'GWNUM': 'GirthWeldNumber',
        'JL.m': 'JointLength_m',
        'RD.m': 'RelativeDistance_m',
        'SO.deg': 'SeamOrientation_deg',
        'SPD.m': 'StartPointDistance_m',
        'SPO.deg': 'StartPointOrientation_deg',
        'EPD.m': 'EndPointDistance_m',
        'EPO.deg': 'EndPointOrientation_deg',
        'SIPRD.M': 'SignificantPointRelDistance_m',
        'SIPO.deg': 'SignificantPointOrientation_deg',
        'WT.mm': 'WallThickness_mm',
        'L.mm': 'FeatureLength_mm',
        'W.mm': 'FeatureWidth_mm',
        'MD.mm': 'MaxDepth_mm'
    }

    # Rename the columns
    df.rename(columns=column_mapping, inplace=True)
    
    return df

class EDA:
    def __init__(self, dataframe):
        self.dataframe = dataframe
    
    def plot_histogram_max_depth(self, column_name='MaxDepth_mm', bins=20, figsize=(8, 4)):
        """
        Plots a histogram of the specified column from the dataframe.
        
        Parameters:
        - column_name: str, name of the column to plot the histogram for
        - bins: int, number of bins for the histogram
        - figsize: tuple, size of the figure
        """
        plt.figure(figsize=figsize)
        plt.hist(self.dataframe[column_name], bins=bins, edgecolor='black')
        plt.xlabel('Max Depth (mm)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Anomaly Depths (Max Depth in mm)')
        plt.show()

    def calculate_percentiles_and_iqr(self, column_name='MaxDepth_mm'):
        """
        Calculates the 25th and 75th percentiles, the IQR, and the lower and upper bounds
        for the specified column from the dataframe.
        
        Parameters:
        - column_name: str, name of the column to calculate the percentiles and IQR for
        """
        Q1 = self.dataframe[column_name].quantile(0.25)
        Q2 = self.dataframe[column_name].quantile(0.50)
        Q3 = self.dataframe[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - 1.5 * IQR)  # Ensure lower bound is not negative
        upper_bound = Q3 + 1.5 * IQR
        
        print(f"25th Percentile (Q1): {Q1}")
        print(f"50th Percentile (Q2): {Q2}")
        print(f"75th Percentile (Q3): {Q3}")
        print(f"Interquartile Range (IQR): {round(IQR, 2)}")
        print(f"Lower Bound: {round(lower_bound, 2)}")
        print(f"Upper Bound: {round(upper_bound, 2)}")

    def plot_boxplot_max_depth(self, column_name='MaxDepth_mm', figsize=(10, 6)):
        """
        Plots a boxplot of the specified column from the dataframe.
        
        Parameters:
        - column_name: str, name of the column to plot the boxplot for
        - figsize: tuple, size of the figure
        """
        plt.figure(figsize=figsize)
        plt.boxplot(self.dataframe[column_name].dropna(), vert=False)
        plt.title('Boxplot of MaxDepth_mm')
        plt.xlabel('MaxDepth_mm')
        plt.show()

    def plot_correlation_matrix(self, figsize=(14, 10), cmap='coolwarm'):
        """
        Plots the correlation matrix of the dataframe.
        
        Parameters:
        - figsize: tuple, size of the figure
        - cmap: str, color map to use for the heatmap
        """
        correlation_matrix = self.dataframe.corr()
        plt.figure(figsize=figsize)
        sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()
    
class MissingValuesAnalyzer:
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def find_missing_values(self):
        # Find the number of missing values in each column
        missing_values = self.dataframe.isna().sum()
        # Print the columns with missing values
        if missing_values.sum() == 0:
            print("No missing values found in 'SeamOrientation_deg' column.")
        else:
            print("Columns with missing values:")
            for column in self.dataframe.columns:
                if missing_values[column] != 0:
                    print(f"{column}: {missing_values[column]}")
    
    def check_inconsistent_seam_orientation(self):
        # Identify the joints where the SeamOrientation_deg is not consistent across the inspection years
        inconsistent_joints = self.dataframe.groupby('GirthWeldNumber').agg({
            'SeamOrientation_deg': lambda x: x.nunique() > 1
        })
        inconsistent_joints = inconsistent_joints[inconsistent_joints['SeamOrientation_deg']].index

        # Check inconsistent joints
        for joint in inconsistent_joints:
            print(f"\nGirthWeldNumber {joint}:")
            joint_data = self.dataframe[self.dataframe['GirthWeldNumber'] == joint]
            for year in joint_data['InspectionYear'].unique():
                year_data = joint_data[joint_data['InspectionYear'] == year]
                print(f"  InspectionYear {year}:")
                print(year_data['SeamOrientation_deg'].value_counts().to_string())
            print("================================================")

    def handle_inconsistent_seam_orientation(self):
        # Identify the joints where the SeamOrientation_deg is not consistent across the inspection years
        inconsistent_joints = self.dataframe.groupby('GirthWeldNumber').agg({
            'SeamOrientation_deg': lambda x: x.nunique() > 1
        })
        inconsistent_joints = inconsistent_joints[inconsistent_joints['SeamOrientation_deg']].index

        # Function to get the most frequent non-null value
        def most_frequent_non_null(series):
            return series.dropna().mode().iloc[0] if not series.dropna().empty else np.nan

        # For each inconsistent joint, apply the most_frequent_non_null function to get the most frequent value
        for joint in inconsistent_joints:
            most_frequent = self.dataframe[self.dataframe['GirthWeldNumber'] == joint]['SeamOrientation_deg'].pipe(most_frequent_non_null)
            
            # Replace all values for this joint with the most frequent value
            self.dataframe.loc[self.dataframe['GirthWeldNumber'] == joint, 'SeamOrientation_deg'] = most_frequent

        # Verify the changes
        for joint in inconsistent_joints:
            print(f"\nGirthWeldNumber {joint}:")
            joint_data = self.dataframe[self.dataframe['GirthWeldNumber'] == joint]
            for year in joint_data['InspectionYear'].unique():
                year_data = joint_data[joint_data['InspectionYear'] == year]
                print(f"  InspectionYear {year}:")
                print(year_data['SeamOrientation_deg'].value_counts().to_string())
            print("================================================")
            
    def find_and_report_inconsistent_joints(self):
        # Find inconsistent joints
        inconsistent_joints = self.dataframe.groupby('GirthWeldNumber').agg({
            'SeamOrientation_deg': lambda x: x.nunique() > 1
        }).reset_index()

        # Filter to keep only the inconsistent joints
        inconsistent_joints = inconsistent_joints[inconsistent_joints['SeamOrientation_deg']]

        if inconsistent_joints.empty:
            print("All joints have consistent seam orientations.")
        else:
            print("The following joints have inconsistent seam orientations:")
            print("\nDetailed breakdown of inconsistent joints by InspectionYear:")
            for joint in inconsistent_joints['GirthWeldNumber']:
                print(f"\nGirthWeldNumber {joint}:")
                joint_data = self.dataframe[self.dataframe['GirthWeldNumber'] == joint]
                
                for year in joint_data['InspectionYear'].unique():
                    year_data = joint_data[joint_data['InspectionYear'] == year]
                    print(f"  InspectionYear {year}:")
                    print(year_data['SeamOrientation_deg'].value_counts().to_string())
                print("================================================")

    def fill_missing_seam_orientation_w_average(self):
        # Fill missing values in 'SeamOrientation_deg' with the mean value of the group 'GirthWeldNumber'
        self.dataframe['SeamOrientation_deg'] = self.dataframe.groupby('GirthWeldNumber')['SeamOrientation_deg'].transform(lambda x: x.fillna(x.mean()))
        return self.dataframe

    def fill_missing_seam_orientation_w_ffill(self):
        # Fill missing values in 'SeamOrientation_deg' by forward filling within each GirthWeldNumber group

        # Step 1: Sort the DataFrame by GirthWeldNumber to ensure correct order
        self.dataframe = self.dataframe.sort_values('GirthWeldNumber')

        # Step 2: Forward fill the SeamOrientation_deg
        self.dataframe['SeamOrientation_deg'] = self.dataframe['SeamOrientation_deg'].ffill()
        return self.dataframe


class FeatureEngineering:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def compute_aspect_ratio(self):
        """
        Compute the aspect ratio of features and handle potential division by zero.
        """
        self.dataframe['AspectRatio'] = self.dataframe['FeatureLength_mm'] / self.dataframe['FeatureWidth_mm']
        self.dataframe['AspectRatio'] = self.dataframe['AspectRatio'].replace([np.inf, -np.inf], np.nan)
        return self.dataframe

    def calculate_feature_area(self):
        """
        Calculate the area of features assuming an elliptical shape.
        """
        self.dataframe['FeatureArea_mm2'] = np.pi * (self.dataframe['FeatureLength_mm'] / 2) * (self.dataframe['FeatureWidth_mm'] / 2)
        return self.dataframe

    def calculate_depth(self, area, length, thickness):
        """
        Calculate depth using the B31G Modified equation (RSTRENG method)
        from ASME B31G standard for "Manual for Determining the Remaining Strength of Corroded Pipelines"
        
        Equation: d = t * (1 - sqrt(A / (L * t)))
        Where:
        d = depth of the corrosion anomaly
        t = nominal wall thickness of the pipe
        A = measured area of metal loss
        L = measured longitudinal extent of the corrosion
        """
        if length == 0 or thickness == 0:
            return 0  # Return 0 if length or thickness is 0 to avoid division by zero
        
        depth = thickness * (1 - np.sqrt(area / (length * thickness)))
        return max(0, min(depth, thickness))  # Ensure depth is between 0 and wall thickness

    def add_estimated_max_depth(self):
        """
        Create the new feature 'estimated_max_depth_mm' using the calculated feature area and depth.
        """
        self.dataframe['estimated_max_depth_mm'] = self.dataframe.apply(
            lambda row: self.calculate_depth(
                row['FeatureArea_mm2'],
                row['FeatureLength_mm'],
                row['WallThickness_mm']
            ),
            axis=1
        )
        return self.dataframe