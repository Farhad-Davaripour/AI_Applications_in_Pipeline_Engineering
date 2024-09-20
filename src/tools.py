import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import glob
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.under_sampling import RandomUnderSampler
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from skopt.callbacks import DeltaYStopper
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator

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
        
    def process_in_increments(self, save_path, increment_size=1000):

        # Find the maximum GirthWeldNumber
        max_girth_weld_number = self.df['GirthWeldNumber'].max()

        # Loop through the data in increments
        for start in range(0, max_girth_weld_number + increment_size, increment_size):
            end = start + increment_size
            subset_df = self.df[(self.df.GirthWeldNumber >= start) & 
                                (self.df.GirthWeldNumber < end)]
            if not subset_df.empty:
                mapper = Anomaly_mapping(subset_df, self.relative_distance_threshold, self.orientation_threshold)
                mapped_subset = mapper.process_all_girth_welds()
                
                # Save the mapped subset to a file
                file_name = os.path.join(save_path, f'Anomaly_mapped_df_{start}_{end}.csv')
                mapped_subset.to_csv(file_name, index=False)
                
                # Reset the results list
                results = []
    
    def concat_mapped_dfs(self, save_path):
        # Define the path to the saved CSV files
        all_files = glob.glob(os.path.join(save_path, "Anomaly_mapped_df_*.csv"))

        # Initialize an empty list to store the dataframes
        dataframes = []

        # Loop through the list of files and read each one into a dataframe
        for file in all_files:
            df = pd.read_csv(file)
            dataframes.append(df)

        # Concatenate all the dataframes into a single dataframe
        Anomaly_mapped_df = pd.concat(dataframes, ignore_index=True)

        # Optionally, you can save this to a CSV file
        Anomaly_mapped_df_file_path = os.path.join(save_path, 'Anomalies_Mapped_All_GirthWelds.csv')
        Anomaly_mapped_df.to_csv(Anomaly_mapped_df_file_path, index=False)

        return Anomaly_mapped_df

def plot_anomalies_by_year(anomalies_df, girth_weld_number, figsize=(15, 6)):
    # Ensure the 'artifacts' directory exists
    os.makedirs('../artifacts', exist_ok=True)
    
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
    
class ErroneousAnomalyProcessor:
    def __init__(self, df):
        self.df = df

    def detect_errors(self, row, length_change_threshold=10, width_change_threshold=50, depth_change_threshold=0.5):
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
        - length_change_threshold = 10 mm: Changes in feature length greater than 10 mm are significant and may indicate an error or substantial anomaly in the pipeline.
        - width_change_threshold = 50 mm: A change greater than 50 mm could signal a substantial change in the feature's shape or an error.
        - depth_change_threshold = 0.5 mm: Depth measurements are critical for assessing the severity of anomalies. A change greater than 0.5 mm is significant and could indicate corrosion or another issue requiring attention.
        """
        if abs(row['LengthChange']) > length_change_threshold:
            return "Error"
        if abs(row['WidthChange']) > width_change_threshold:
            return "Error"
        if abs(row['DepthChange']) > depth_change_threshold:
            return "Error"
        return "Okay"

    def print_error_statistics(self):
        """
        Prints statistics about the erroneous and correct records.
        """
        ErrorClassification_true = self.df[self.df.ErrorClassification == 'Error']
        ErrorClassification_false = self.df[self.df.ErrorClassification == 'Okay']

        print(f"number of erroneous records: {len(ErrorClassification_true)}")
        print(f"number of correct records:  {len(ErrorClassification_false)}")
        print(f"percentage of erroneous records: {len(ErrorClassification_true) / len(self.df) * 100:.2f}%\n")

        old_records = self.df[self.df.Tag == 'old']
        new_records = self.df[self.df.Tag == 'new']

        print(f"number of old records: {len(old_records)}")
        print(f"number of new records: {len(new_records)}")
        print(f"percentage of old records: {len(old_records) / len(self.df) * 100:.2f}%")

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

        # Variable to store the last inconsistent joint's data
        last_inconsistent_joint_data = None

        # Check inconsistent joints
        for joint in inconsistent_joints:
            joint_data = self.dataframe[self.dataframe['GirthWeldNumber'] == joint]
            joint_details = f"\nGirthWeldNumber {joint}:\n"
            for year in joint_data['InspectionYear'].unique():
                year_data = joint_data[joint_data['InspectionYear'] == year]
                joint_details += f"  InspectionYear {year}:\n"
                joint_details += year_data['SeamOrientation_deg'].value_counts().to_string() + "\n"
            joint_details += "================================================\n"
            
            # Store the details of the current joint
            last_inconsistent_joint_data = joint_details

        # Print the last inconsistent joint's details
        if last_inconsistent_joint_data:
            print(f"The last inconsistent joint's details:\n {last_inconsistent_joint_data}")

    def handle_inconsistent_seam_orientation(self):
        # Identify the joints where the SeamOrientation_deg is not consistent across the inspection years
        inconsistent_joints = self.dataframe.groupby('GirthWeldNumber').agg({
            'SeamOrientation_deg': lambda x: x.nunique() > 1
        })
        inconsistent_joints = inconsistent_joints[inconsistent_joints['SeamOrientation_deg']].index

        # Function to get the most frequent non-null value
        def most_frequent_non_null(series):
            return series.dropna().mode().iloc[0] if not series.dropna().empty else np.nan

        # Variable to store the last inconsistent joint's data
        last_inconsistent_joint_data = None

        # For each inconsistent joint, apply the most_frequent_non_null function to get the most frequent value
        for joint in inconsistent_joints:
            most_frequent = self.dataframe[self.dataframe['GirthWeldNumber'] == joint]['SeamOrientation_deg'].pipe(most_frequent_non_null)
            
            # Replace all values for this joint with the most frequent value
            self.dataframe.loc[self.dataframe['GirthWeldNumber'] == joint, 'SeamOrientation_deg'] = most_frequent

        # Verify the changes and store the last inconsistent joint's data
        for joint in inconsistent_joints:
            joint_data = self.dataframe[self.dataframe['GirthWeldNumber'] == joint]
            joint_details = f"\nGirthWeldNumber {joint}:\n"
            for year in joint_data['InspectionYear'].unique():
                year_data = joint_data[joint_data['InspectionYear'] == year]
                joint_details += f"  InspectionYear {year}:\n"
                joint_details += year_data['SeamOrientation_deg'].value_counts().to_string() + "\n"
            joint_details += "================================================\n"
            
            # Store the details of the current joint
            last_inconsistent_joint_data = joint_details

        # Print the last inconsistent joint's details
        if last_inconsistent_joint_data:
            print(last_inconsistent_joint_data)
            
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
        # Filling missing values in 'SeamOrientation_deg' with the mean value of the group 'GirthWeldNumber'
        original_missing = self.dataframe['SeamOrientation_deg'].isna().sum()
        self.dataframe['SeamOrientation_deg'] = self.dataframe.groupby('GirthWeldNumber')['SeamOrientation_deg'].transform(lambda x: x.fillna(x.mean()))
        filled = original_missing - self.dataframe['SeamOrientation_deg'].isna().sum()
        
        # Return the number of values that were filled with the group's mean
        return filled

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
    
    @staticmethod
    def _deg_to_rad(deg):
        """
        Convert degrees to radians.
        """
        return deg * np.pi / 180
    
    def add_angular_features(self, angle_columns=None):
        """
        Create sine and cosine features for each angular measurement.
        """
        angle_columns = angle_columns
        
        for col in angle_columns:
            # Convert to radians
            self.dataframe[f'{col}_rad'] = self.dataframe[col].apply(self._deg_to_rad)
            
            # Create sine component
            self.dataframe[f'{col}_sin'] = np.sin(self.dataframe[f'{col}_rad'])
            
            # Create cosine component
            self.dataframe[f'{col}_cos'] = np.cos(self.dataframe[f'{col}_rad'])
        
        return self.dataframe
    
def add_dprev_features(df):
    """
    Adds second previous inspection year features (DPrev) to the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame with anomaly data.

    Returns:
    pd.DataFrame: The DataFrame with added DPrev features.
    """
    DPrev_Old_Filtered_Anomaly_mapped_df = df.copy()

    # Initialize new columns for DPrev values in the copied DataFrame
    DPrev_Old_Filtered_Anomaly_mapped_df['DPrev_RelativeDistance_m'] = None
    DPrev_Old_Filtered_Anomaly_mapped_df['DPrev_FeatureLength_mm'] = None
    DPrev_Old_Filtered_Anomaly_mapped_df['DPrev_FeatureWidth_mm'] = None
    DPrev_Old_Filtered_Anomaly_mapped_df['DPrev_MaxDepth_mm'] = None
    DPrev_Old_Filtered_Anomaly_mapped_df['DPrev_SignificantPointOrientation_deg'] = None

    # Iterate through each row to find and assign DPrev values
    total_rows = len(DPrev_Old_Filtered_Anomaly_mapped_df)
    for index, row in DPrev_Old_Filtered_Anomaly_mapped_df.iterrows():
        # Get the current row's Prev values
        prev_relative_distance = row['Prev_RelativeDistance_m']
        prev_depth = row['Prev_MaxDepth_mm']
        prev_feature_length = row['Prev_FeatureLength_mm']
        prev_feature_width = row['Prev_FeatureWidth_mm']
        prev_orientation = row['Prev_SignificantPointOrientation_deg']
        
        # Find the row with matching RelativeDistance_m, MaxDepth_mm, FeatureLength_mm, FeatureWidth_mm, and SignificantPointOrientation_deg
        matching_row = DPrev_Old_Filtered_Anomaly_mapped_df[
            (DPrev_Old_Filtered_Anomaly_mapped_df['RelativeDistance_m'] == prev_relative_distance) & 
            (DPrev_Old_Filtered_Anomaly_mapped_df['MaxDepth_mm'] == prev_depth) &
            (DPrev_Old_Filtered_Anomaly_mapped_df['FeatureLength_mm'] == prev_feature_length) &
            (DPrev_Old_Filtered_Anomaly_mapped_df['FeatureWidth_mm'] == prev_feature_width) &
            (DPrev_Old_Filtered_Anomaly_mapped_df['SignificantPointOrientation_deg'] == prev_orientation)
        ]
        
        if not matching_row.empty:
            # Retrieve the corresponding Prev values from the matching row
            dprev_relative_distance = matching_row['Prev_RelativeDistance_m'].values[0]
            dprev_feature_length = matching_row['Prev_FeatureLength_mm'].values[0]
            dprev_feature_width = matching_row['Prev_FeatureWidth_mm'].values[0]
            dprev_max_depth = matching_row['Prev_MaxDepth_mm'].values[0]
            dprev_orientation = matching_row['Prev_SignificantPointOrientation_deg'].values[0]
            
            # Assign these DPrev values to the original row
            DPrev_Old_Filtered_Anomaly_mapped_df.at[index, 'DPrev_RelativeDistance_m'] = dprev_relative_distance
            DPrev_Old_Filtered_Anomaly_mapped_df.at[index, 'DPrev_FeatureLength_mm'] = dprev_feature_length
            DPrev_Old_Filtered_Anomaly_mapped_df.at[index, 'DPrev_FeatureWidth_mm'] = dprev_feature_width
            DPrev_Old_Filtered_Anomaly_mapped_df.at[index, 'DPrev_MaxDepth_mm'] = dprev_max_depth
            DPrev_Old_Filtered_Anomaly_mapped_df.at[index, 'DPrev_SignificantPointOrientation_deg'] = dprev_orientation

        # Print progress
        if index % 100 == 0 or index == total_rows - 1:
            print(f"Processed {index + 1} / {total_rows} rows")

    return DPrev_Old_Filtered_Anomaly_mapped_df

class HandlingOutlier:
    def __init__(self, df):
        self.df = df

    def remove_outliers_zscore(self, columns, threshold=3):
        """
        Removes outliers based on Z-scores for the specified columns.

        Parameters:
        columns (list): List of column names to calculate Z-scores.
        threshold (float): Z-score threshold to identify outliers. Default is 3.
        
        Returns:
        pd.DataFrame: DataFrame with outliers removed.
        """
        z_scores = np.abs(zscore(self.df[columns], nan_policy='omit'))
        mask = (z_scores < threshold).all(axis=1)
        self.df = self.df[mask]
        return self.df

    def remove_outliers_isolation_forest(self, columns, contamination=0.05):
        """
        Removes outliers using Isolation Forest for the specified columns.

        Parameters:
        columns (list): List of column names to apply Isolation Forest.
        contamination (float): Proportion of outliers in the data set. Default is 0.05.
        
        Returns:
        pd.DataFrame: DataFrame with outliers removed.
        """
        iso_forest = IsolationForest(contamination=contamination)
        outliers = iso_forest.fit_predict(self.df[columns])
        self.df = self.df[outliers == 1]
        return self.df
    
class FeatureImportance:
    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.scaler = StandardScaler()
        self.best_lasso = None
        self.importance_df = None

    def standardize_features(self):
        self.features_scaled = self.scaler.fit_transform(self.features)
        print("Standardization of features is done.")

    def split_data(self, test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features_scaled, self.target, test_size=test_size, random_state=random_state
        )
        print("Data splitting into training and testing sets is done.")

    def perform_grid_search(self):
        alpha_grid = {'alpha': np.linspace(0.001, 1, 100)} 
        lasso = Lasso()
        grid_search = GridSearchCV(estimator=lasso, param_grid=alpha_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(self.X_train, self.y_train)
        self.best_alpha = grid_search.best_params_['alpha']
        print(f"Grid search is done. Best alpha value: {self.best_alpha}")

    def fit_best_lasso(self):
        self.best_lasso = Lasso(alpha=self.best_alpha)
        self.best_lasso.fit(self.X_train, self.y_train)
        print("Fitting the Lasso model with the best alpha is done.")

    def calculate_coefficients(self):
        coefficients = self.best_lasso.coef_
        self.importance_df = pd.DataFrame({'Feature': self.features.columns, 'Coefficient': coefficients})
        self.importance_df.sort_values(by='Coefficient', ascending=False, inplace=True)
        print("Calculation of feature coefficients is done.")

    def plot_coefficients(self):
        plt.figure(figsize=(6, 12))
        sns.barplot(x='Coefficient', y='Feature', data=self.importance_df)
        plt.title('Feature Importance using Lasso Regression with Best Alpha')
        plt.show()
        print("Plotting of coefficients is done.")

    def plot_non_zero_coefficients(self):
        non_zero_importance = self.importance_df[self.importance_df['Coefficient'] != 0]
        plt.figure(figsize=(6, 12))
        sns.barplot(x='Coefficient', y='Feature', data=non_zero_importance)
        plt.title('Non-zero Feature Importance using Lasso Regression with Best Alpha')
        plt.show()
        print("Plotting of non-zero coefficients is done.")

class TrainingPipeline:

    def __init__(self, features, target):
        self.features = features
        self.target = target
        self.scaler = StandardScaler()
        self.features_scaled = None
        self.features_resampled = None
        self.target_resampled = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model = None

    def scale_features(self):
        self.features = self.features.loc[self.target.index]
        self.features_scaled = self.scaler.fit_transform(self.features)
        return self.features_scaled

    def handle_class_imbalance(self):
        original_discretized_values = np.unique(self.target)
        n_bins = len(original_discretized_values)
        kbins = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
        target_binned = kbins.fit_transform(self.target.values.reshape(-1, 1)).reshape(-1).astype(int)
        bin_counts = np.bincount(target_binned)
        median_freq = np.median(bin_counts)
        sampling_strategy = {i: min(count, int(median_freq)) for i, count in enumerate(bin_counts) if count > 0}
        rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        self.features_resampled, target_resampled_binned = rus.fit_resample(self.features, target_binned)
        self.target_resampled = original_discretized_values[target_resampled_binned]
        return self.features_resampled, self.target_resampled

    def split_data(self, test_size=0.2, random_state=42, handle_imbalance=False):
        if handle_imbalance:
            self.handle_class_imbalance()
            X = self.features_resampled
            y = self.target_resampled
        else:
            X = self.features_scaled
            y = self.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        return self.X_train, self.X_test, self.y_train, self.y_test

    def hyperparameter_tuning(self, search_spaces=None, n_iter=50, cv=5, scoring='neg_mean_squared_error'):
        if search_spaces is None:
            search_spaces = {
                'learning_rate': Real(0.01, 0.5, prior='log-uniform'),
                'max_iter': Integer(100, 500),
                'max_depth': Integer(3, 10),
                'min_samples_leaf': Integer(5, 50),
                'l2_regularization': Real(1e-6, 1e-2, prior='log-uniform')
            }
        base_model = HistGradientBoostingRegressor(random_state=42)
        delta_y_stopper = DeltaYStopper(delta=0.001)
        bayes_search = BayesSearchCV(
            base_model,
            search_spaces,
            n_iter=n_iter,
            cv=cv,
            n_jobs=-1,
            random_state=42,
            scoring=scoring)
        bayes_search.fit(self.X_train, self.y_train, callback=[delta_y_stopper])
        best_params = bayes_search.best_params_
        self.best_model = HistGradientBoostingRegressor(random_state=42, **best_params)
        return best_params

    def fit_model(self, use_best_params=True):
        if self.best_model is None or not use_best_params:
            self.best_model = HistGradientBoostingRegressor(
                l2_regularization=2.148188547134838e-06,
                learning_rate=0.01,
                max_depth=3,
                max_iter=386,
                min_samples_leaf=5,
                random_state=42
            )
        self.best_model.fit(self.X_train, self.y_train)
        return self.best_model

    def evaluate_model(self, min_value=None, max_value=None):
        # Filter y_test based on the min_value and max_value if provided
        mask = np.ones(len(self.y_test), dtype=bool)  # Initialize mask with all True
        if min_value is not None:
            mask = mask & (self.y_test >= min_value)
        if max_value is not None:
            mask = mask & (self.y_test <= max_value)
        
        y_test_filtered = self.y_test[mask]
        X_test_filtered = self.X_test[mask]

        # Predict using the filtered X_test
        y_pred = self.best_model.predict(X_test_filtered)

        # Evaluate performance metrics on the filtered data
        mse = mean_squared_error(y_test_filtered, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_filtered, y_pred)
        r2 = r2_score(y_test_filtered, y_pred)
        mape = np.mean(np.abs((y_test_filtered - y_pred) / y_test_filtered)) * 100
        me = np.mean(y_test_filtered - y_pred)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'ME': me
        }
    
    def plot_prediction_accuracy(self):
        y_pred = self.best_model.predict(self.X_test)
        results = pd.DataFrame({'Actual': self.y_test, 'Predicted': y_pred})

        tolerances = np.arange(0.01, 0.51, 0.01)
        accuracies = []
        for tolerance in tolerances:
            accuracy = np.mean(np.abs((self.y_test - y_pred) / self.y_test) < tolerance) * 100
            accuracies.append(accuracy)

        plt.figure(figsize=(6, 4))
        plt.plot(tolerances * 100, accuracies, marker='o')
        plt.xlabel('Tolerance (%)')
        plt.ylabel('Percentage of Predictions Within Tolerance (%)')
        plt.title('Prediction Accuracy for Different Tolerance Ranges')
        plt.grid(True)
        plt.show()

        return results

    def plot_violin(self, results, figsize=(15, 6)):
        plt.figure(figsize=figsize)
        sns.violinplot(x='Actual', y='Predicted', data=results)

        plt.xlabel('Max Depth (mm)')
        plt.ylabel('Predicted Max Depth (mm)')
        plt.title('Actual vs Predicted Values')
        plt.show()

    def plot_scatter(self, results):
        sns.scatterplot(x='Actual', y='Predicted', data=results, color='blue', alpha=0.5)

        min_val = min(results['Actual'].min(), results['Predicted'].min())
        max_val = max(results['Actual'].max(), results['Predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

        plt.xlabel('Max Depth (mm)')
        plt.ylabel('Predicted Max Depth (mm)')
        plt.title('Actual vs Predicted Values')
        plt.show()

    def fill_missing_values(self, dataframe, feature_columns, target_column):
        missing_index = dataframe[target_column].isnull()
        dataframe.loc[missing_index, f'{target_column}_predicted'] = self.best_model.predict(self.scaler.transform(dataframe.loc[missing_index, feature_columns]))

        missing_values_remaining = dataframe[target_column].isnull().sum()
        print(f"Number of remaining missing values in '{target_column}': {missing_values_remaining}")

        predicted_records = dataframe[dataframe[target_column].isnull()]
        return predicted_records

class AnomalyClusterer:
    def __init__(self, dataframe, clustering_features, n_clusters=5, random_state=42):
        self.dataframe = dataframe.copy()  # Make a copy to avoid modifying the original DataFrame
        self.clustering_features = clustering_features
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()  # Initialize the scaler here
        self.features_normalized = None
        self.principal_components = None
        self.pca = None

    def perform_clustering(self):
        # Select the features for clustering
        features_kmeans = self.dataframe[self.clustering_features]

        # Normalize the data
        self.features_normalized = self.scaler.fit_transform(features_kmeans)

        # Fit a KMeans model
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(self.features_normalized)

        # Assign cluster labels to the data
        self.dataframe['anomaly_type'] = self.kmeans.labels_

        # Perform PCA to reduce dimensionality to 2D for visualization
        self.pca = PCA(n_components=7)
        self.principal_components = self.pca.fit_transform(self.features_normalized)

        # Add principal components to the original dataframe
        self.dataframe['PC1'] = self.principal_components[:, 0]
        self.dataframe['PC2'] = self.principal_components[:, 1]

        return self.dataframe

    def visualize_clusters(self):
        # Ensure that perform_clustering has been called
        if self.principal_components is None:
            raise ValueError("You need to call perform_clustering() before visualize_clusters().")

        # Plot the clusters
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(self.dataframe['PC1'], self.dataframe['PC2'],
                              c=self.dataframe['anomaly_type'], cmap='viridis')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of Anomaly Clusters')
        plt.legend(*scatter.legend_elements(), title="Clusters")
        plt.show()

    def plot_pca_explained_variance(self):
        if self.pca is None:
            raise ValueError("You need to call perform_clustering() before plotting PCA explained variance.")

        # Get the explained variance ratios and compute cumulative variance
        explained_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        components = np.arange(1, len(explained_variance_ratio) + 1)

        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.bar(components, explained_variance_ratio, alpha=0.7, label='Individual Explained Variance')
        plt.step(components, cumulative_variance, where='mid', color='red', label='Cumulative Explained Variance')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance Ratio by Component')
        plt.xticks(components)
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()

class AnomalyPredictionPipeline:
    def __init__(self, df, model: BaseEstimator, prev_inspection_year: int, next_inspection_year: int):
        self.df = df
        self.model = model
        self.prev_inspection_year = prev_inspection_year
        self.next_inspection_year = next_inspection_year

    def prepare_data(self, wt_mm: pd.Series, feature_columns: list, target_column: str):
        # Filtering data for the previous inspection year
        self.df = self.df[self.df['InspectionYear'] == self.prev_inspection_year].copy()
        self.df[target_column] = self.df[self.df['InspectionYear'] == self.prev_inspection_year][target_column]

        # Assigning the new inspection year and the previous inspection year
        self.df['InspectionYear'] = self.next_inspection_year
        self.df['Prev_InspectionYear'] = self.prev_inspection_year
        self.df['WallThickness_mm'] = wt_mm

        # Creating new columns with the previous values
        for col in ['RelativeDistance_m', 'FeatureLength_mm', 'FeatureWidth_mm', 'MaxDepth_mm', 'SignificantPointOrientation_deg']:
            self.df[f'Prev_{col}'] = self.df[col]

        # Delta values for the previous inspection year
        for col in ['RelativeDistance_m', 'FeatureLength_mm', 'FeatureWidth_mm', 'MaxDepth_mm', 'SignificantPointOrientation_deg']:
            self.df[f'DPrev_{col}'] = self.df[f'Prev_{col}']

        # Derived feature calculations
        self.df['Estimated_FeatureLength_mm'] = (
            2 * self.df['FeatureLength_mm'] - self.df['Prev_FeatureLength_mm']
        )
        self.df['Estimated_FeatureWidth_mm'] = (
            2 * self.df['FeatureWidth_mm'] - self.df['Prev_FeatureWidth_mm']
        )
        self.df['Powered_Prev_MaxDepth_mm'] = (
            self.df['MaxDepth_mm'] ** 2
        )

        # Adding a sanity check to ensure all columns are present
        required_columns = feature_columns + [target_column] + ['WallThickness_mm']
        missing_columns = [col for col in required_columns if col not in self.df.columns]

        if missing_columns:
            print(f"Warning: The following required columns are missing: {missing_columns}")

        # Ensure the final DataFrame has all the required columns
        self.df = self.df[required_columns]
        
        print("Data preparation is done.")
        return self.df

    def make_predictions(self, feature_columns: list):
        # Make predictions
        self.df['Prediction_MaxDepth_mm'] = self.model.predict(self.df[feature_columns])
        
        # Ensure predicted max depth is not smaller than actual max depth
        self.df['Prediction_MaxDepth_mm'] = self.df.apply(
            lambda row: max(row['Prediction_MaxDepth_mm'], row['MaxDepth_mm']),
            axis=1
        )

        print("Predictions are done.")
        return self.df


    def perform_analytics(self, figsize=(10, 6)):
        # Calculate the total and count of predicted max depths, mean, and standard deviation for each GirthWeldNumber
        grouped_df = self.df.groupby('GirthWeldNumber').agg(
            total_pred_max_depth=('Prediction_MaxDepth_mm', 'sum'),
            mean_pred_max_depth=('Prediction_MaxDepth_mm', 'mean'),
            std_pred_max_depth=('Prediction_MaxDepth_mm', 'std'),
            total_max_depth=('MaxDepth_mm', 'sum'),
            mean_max_depth=('MaxDepth_mm', 'mean'),
            count=('Prediction_MaxDepth_mm', 'size'),
            wall_thickness=('WallThickness_mm', 'mean')  # Added wall thickness
        )

        # Handle cases where std_pred_max_depth might be NaN (e.g., if count is 1)
        grouped_df['std_pred_max_depth'].fillna(0, inplace=True)

        # Calculate the weighted density (average predicted depth)
        grouped_df['weighted_density_division'] = grouped_df['total_pred_max_depth'] / grouped_df['count']

        # Calculate the total impact (criticality)
        grouped_df['criticality_multiplication'] = grouped_df['total_pred_max_depth'] * grouped_df['count']

        # Normalize the criticality_multiplication values
        min_criticality = grouped_df['criticality_multiplication'].min()
        max_criticality = grouped_df['criticality_multiplication'].max()
        grouped_df['normalized_criticality_multiplication'] = (
            (grouped_df['criticality_multiplication'] - min_criticality) / (max_criticality - min_criticality)
        )

        # Sort by weighted density from maximum to minimum
        sorted_by_density_division = grouped_df.sort_values(by='weighted_density_division', ascending=False).head(10)

        # Sort by normalized criticality from maximum to minimum
        sorted_by_criticality_multiplication = grouped_df.sort_values(by='normalized_criticality_multiplication', ascending=False).head(10)

        # Print top 10 anomalies with the most depth including predicted max depth
        top10_anomalies = self.df.sort_values(by='Prediction_MaxDepth_mm', ascending=False).head(10)
        print("\nTop 10 Anomalies with Most Predicted Depth:")
        print(top10_anomalies[['GirthWeldNumber', 'MaxDepth_mm', 'Prediction_MaxDepth_mm', 'WallThickness_mm']])

        # Print top 10 joints with the highest sum of predicted anomaly depth
        top10_joints_by_sum_pred_depth = grouped_df.sort_values(by='total_pred_max_depth', ascending=False).head(10)
        print("\nTop 10 Joints with Highest Sum of Predicted Anomaly Depth:")
        print(top10_joints_by_sum_pred_depth[['total_max_depth', 'total_pred_max_depth', 'mean_max_depth', 'count', 'mean_pred_max_depth', 'wall_thickness']])

        # Plot top 10 GirthWeldNumbers by weighted density (division)
        plt.figure(figsize=figsize)
        plt.bar(sorted_by_density_division.index.astype(str), sorted_by_density_division['weighted_density_division'])
        plt.xlabel('Joint Number')
        plt.ylabel('Average Predicted Depth')
        plt.title('Top 10 Joints by Average Predicted Anomaly Depth')
        plt.xticks(rotation=45)
        plt.show()

        # Plot top 10 GirthWeldNumbers by normalized criticality (multiplication)
        plt.figure(figsize=figsize)
        plt.bar(sorted_by_criticality_multiplication.index.astype(str), sorted_by_criticality_multiplication['normalized_criticality_multiplication'])
        plt.xlabel('Joint Number')
        plt.ylabel('Normalized Total Predicted Impact')
        plt.title('Top 10 Joints by Normalized Total Predicted Anomaly Impact')
        plt.xticks(rotation=45)
        plt.show()

        print("Analytics are done.")

