import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from logger import setup_logger
import glob
import argparse

# Configure logger
logger = setup_logger("Plot data")

def load_anomalies_file(file_path=None):
    """
    Load anomalies data from a CSV file.
    
    Parameters:
        file_path: Path to the anomalies CSV file. If None, uses the most recent file in the anomalies directory.
        
    Returns:
        DataFrame containing anomaly data
    """
    if file_path is None:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        anomalies_dir = os.path.join(current_dir, "anomalies")
        
        # Check if anomalies directory exists
        if not os.path.exists(anomalies_dir):
            logger.error(f"Error: The anomalies directory does not exist at {anomalies_dir}")
            return None
        
        # Get all CSV files in the anomalies directory
        csv_files = glob.glob(os.path.join(anomalies_dir, "*.csv"))
        
        if not csv_files:
            logger.error("Error: No CSV files found in the anomalies directory")
            return None
        
        # Get the most recent anomalies file
        file_path = max(csv_files, key=os.path.getctime)
        logger.info(f"Using most recent anomalies file: {file_path}")
    
    try:
        # Read the anomalies file
        anomalies_df = pd.read_csv(file_path)
        
        # If 'start_date' column doesn't exist, create it based on filename or current date
        if 'start_date' not in anomalies_df.columns:
            # Try to extract date from filename (e.g., anomalies_2025_03_31.csv)
            match = re.search(r'(\d{4})_(\d{2})_(\d{2})', os.path.basename(file_path))
            if match:
                year, month, day = match.groups()
                date_str = f"{day}/{month}/{year}"
                anomalies_df['start_date'] = date_str
            else:
                # Use current date as fallback
                today = datetime.now().strftime("%d/%m/%Y")
                anomalies_df['start_date'] = today
        
        # Convert date column to datetime if it exists
        if 'start_date' in anomalies_df.columns:
            anomalies_df['start_date'] = pd.to_datetime(anomalies_df['start_date'], format='%d/%m/%Y', errors='coerce')
        
        return anomalies_df
    
    except Exception as e:
        logger.error(f"Error reading anomalies file {file_path}: {e}")
        return None

def get_historical_data_for_anomalies(anomalies_df, metric_name='cost'):
    """
    Query the database to get historical data for the anomalies.
    If querying fails, create a synthetic dataset based on the anomalies data.
    
    Parameters:
        anomalies_df: DataFrame containing anomaly data
        metric_name: Name of the metric column
        
    Returns:
        DataFrame containing historical data
    """
    import subprocess
    from io import StringIO
    
    # Create a list to store the historical data
    historical_data_frames = []
    
    # First try to get data from the database
    try:
        # Get unique combinations of service and subscription
        service_sub_combinations = anomalies_df[['service', 'subscription_name']].drop_duplicates()
        
        for _, row in service_sub_combinations.iterrows():
            service = row['service']
            subscription = row['subscription_name']
            
            # Extract subscription ID from subscription name if possible
            subscription_id = ""
            match = re.search(r'\((.*?)\)', subscription)
            if match:
                subscription_id = match.group(1)
                subscription_condition = f"subscription_id = '{subscription_id}'"
            else:
                # If no ID in parentheses, try to match by subscription name
                subscription_condition = f"subscription_name LIKE '%{subscription}%'"
            
            # Build SQL query to get historical data
            query = f"""
            SELECT service, subscription_name, start_date, cost
            FROM df
            WHERE service = '{service}' AND {subscription_condition}
            ORDER BY start_date
            """
            
            # Run the SQL query using sql_df.py
            try:
                cmd = f"python sql_df.py --query \"{query}\""
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                
                if result.returncode == 0:
                    # Parse the output
                    output_lines = result.stdout.strip().split('\n')
                    data_lines = []
                    capture = False
                    
                    for line in output_lines:
                        if line.startswith('Query Results:'):
                            capture = True
                        elif capture and line.strip():
                            data_lines.append(line)
                    
                    if data_lines:
                        # Convert to DataFrame
                        data_str = '\n'.join(data_lines)
                        df = pd.read_csv(StringIO(data_str), sep=r'\s+')
                        historical_data_frames.append(df)
            except Exception as e:
                logger.error(f"Error executing SQL query: {e}")
    except Exception as e:
        logger.error(f"Error getting historical data from database: {e}")
    
    # If we couldn't get historical data from the database, create synthetic data
    if not historical_data_frames:
        logger.warning("Creating synthetic historical data based on anomalies")
        
        # Get today's date
        today = pd.Timestamp.now().normalize()
        
        # Create synthetic historical data for each anomaly
        for _, row in anomalies_df.iterrows():
            service = row['service']
            subscription = row['subscription_name']
            
            # Use avg_historical_cost if available, otherwise use yesterday_cost
            if 'avg_historical_cost' in row and pd.notna(row['avg_historical_cost']):
                avg_cost = row['avg_historical_cost']
            else:
                avg_cost = row[metric_name] * 0.8  # Assume 80% of current cost as historical average
            
            # Create 90 days of historical data with some random variation
            dates = [today - pd.Timedelta(days=i) for i in range(90, 0, -1)]
            
            # Create a DataFrame with synthetic data
            synthetic_data = []
            for date in dates:
                # Add some random variation (±20%)
                random_factor = 0.8 + (np.random.random() * 0.4)  # Between 0.8 and 1.2
                cost_value = avg_cost * random_factor
                
                synthetic_data.append({
                    'service': service,
                    'subscription_name': subscription,
                    'start_date': date,
                    'cost': cost_value,
                })
            
            # Add today's data point using yesterday_cost
            synthetic_data.append({
                'service': service,
                'subscription_name': subscription,
                'start_date': today,
                'cost': row[metric_name],
            })
            
            # Convert to DataFrame
            df = pd.DataFrame(synthetic_data)
            historical_data_frames.append(df)
    
    if historical_data_frames:
        # Combine all historical data
        historical_df = pd.concat(historical_data_frames, ignore_index=True)
        
        # Convert date column to datetime
        if 'start_date' in historical_df.columns:
            historical_df['start_date'] = pd.to_datetime(historical_df['start_date'], errors='coerce')
        
        return historical_df
    
    return None

def map_anomalies_to_historical(historical_df, anomalies_df, metric_name='cost'):
    """
    Map anomalies data to historical data format to ensure compatibility.
    
    Parameters:
        historical_df: DataFrame containing historical data
        anomalies_df: DataFrame containing anomaly data
        metric_name: Name of the metric column in the anomalies data
        
    Returns:
        DataFrame containing mapped anomaly data
    """
    mapped_anomalies = anomalies_df.copy()
    
    # If the metric name in anomalies is different from 'cost', rename it
    if metric_name != 'cost' and metric_name in mapped_anomalies.columns:
        mapped_anomalies = mapped_anomalies.rename(columns={metric_name: 'cost'})
    
    # If 'resource_name' or 'resource_group' is missing but we need it for plotting, add empty columns
    if 'resource_name' in historical_df.columns and 'resource_name' not in mapped_anomalies.columns:
        mapped_anomalies['resource_name'] = ''
        
    if 'resource_group' in historical_df.columns and 'resource_group' not in mapped_anomalies.columns:
        mapped_anomalies['resource_group'] = ''
    
    return mapped_anomalies

def generate_anomaly_graphs(anomalies_file=None, output_dir=None):
    """
    Generate graphs for anomalies data.
    
    Parameters:
        anomalies_file: Path to the anomalies CSV file. If None, uses the most recent file in the anomalies directory.
        output_dir: Directory to save the generated graphs. If None, creates an 'outputs' directory.
    """
    # Load anomalies data
    anomalies_df = load_anomalies_file(anomalies_file)
    if anomalies_df is None:
        logger.error("Failed to load anomalies data")
        return
    
    # Determine the metric name to use
    metric_name = 'cost'
    if metric_name not in anomalies_df.columns and 'yesterday_cost' in anomalies_df.columns:
        logger.info("Using 'yesterday_cost' as the metric name")
        metric_name = 'yesterday_cost'
    elif metric_name not in anomalies_df.columns:
        logger.error(f"Neither 'cost' nor 'yesterday_cost' columns found in the anomalies file")
        return
    
    # Set up output directory
    if output_dir is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(current_dir, "outputs")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a dated subfolder for this run if we have a date in the anomalies
    date_str = None
    anomalies_date = None
    if 'start_date' in anomalies_df.columns and not anomalies_df['start_date'].empty:
        anomalies_date = anomalies_df['start_date'].iloc[0]
        if anomalies_date:
            date_str = anomalies_date.strftime("%Y_%m_%d")
    
    # If we can't determine the date from the data, try from the filename
    if anomalies_file and not date_str:
        match = re.search(r'(\d{4})_(\d{2})_(\d{2})', os.path.basename(anomalies_file))
        if match:
            date_str = f"{match.group(1)}_{match.group(2)}_{match.group(3)}"
    
    # Final fallback: use today's date
    if not date_str:
        date_str = datetime.now().strftime("%Y_%m_%d")
    
    # Create dated subfolder
    dated_output_dir = os.path.join(output_dir, f"plots_{date_str}")
    os.makedirs(dated_output_dir, exist_ok=True)
    
    # Create timeline_anomalies directory
    timeline_dir = os.path.join(dated_output_dir, "timeline_anomalies")
    os.makedirs(timeline_dir, exist_ok=True)
    
    # Get historical data for the anomalies
    historical_df = get_historical_data_for_anomalies(anomalies_df, metric_name=metric_name)
    
    if historical_df is None or historical_df.empty:
        logger.error("Failed to retrieve historical data for anomalies")
        return
    
    # Generate timeline anomalies graphs
    try:
        logger.info("Generating timeline anomalies graphs...")
        # Map anomalies to the format expected by the timeline function
        mapped_anomalies = map_anomalies_to_historical(historical_df, anomalies_df, metric_name=metric_name)
        plot_timeline_anomalies(historical_df, mapped_anomalies, timeline_dir, anomalies_date)
    except Exception as e:
        logger.error(f"Error generating timeline anomalies graphs: {e}")
    
    logger.info(f"Anomaly graphs generated successfully in {dated_output_dir}")

def plot_timeline_anomalies(grouped_data: pd.DataFrame, anomalies_df: pd.DataFrame,
                            output_folder: str, anomalies_date: datetime = None, 
                            group_by: str = None, metric_name: str = 'cost', 
                            trend_line: bool = True):
    """
    Create timeline line graphs showing cost trends and anomalies over time.
    
    Parameters:
        grouped_data: DataFrame containing all data points
        anomalies_df: DataFrame containing only anomaly data points
        output_folder: Folder where plots will be saved
        anomalies_date: The date of anomalies to highlight in the plot
        group_by: Column name to group anomalies by (e.g., 'resource_name', 'resource_group', etc.)
                 If None, will use the default grouping by subscription only
    """
    # Before modifying the DataFrames, create explicit copies
    grouped_data = grouped_data.copy()
    anomalies_df = anomalies_df.copy()
    
    grouped_data['subscription_name'] = grouped_data['subscription_name'].apply(lambda x: re.match(r'^([^(]+)', x).group(1).strip())
    anomalies_df['subscription_name'] = anomalies_df['subscription_name'].apply(lambda x: re.match(r'^([^(]+)', x).group(1).strip())
    
    # Create timeline folder
    timeline_folder = os.path.join(output_folder, 'timeline_anomalies')
    os.makedirs(timeline_folder, exist_ok=True)

    # Check if the group_by column exists in the dataframe
    has_group_by = group_by is not None and group_by in anomalies_df.columns

    # Check if resource_name or resource_group is in the dataframe
    if 'resource_name' in anomalies_df.columns and not has_group_by:
        has_group_by = True
        group_by = 'resource_name'
        logger.warning("group_by was not provided but resource_name is in the dataframe, using resource_name...")
    elif 'resource_group' in anomalies_df.columns and not has_group_by:
        has_group_by = True
        group_by = 'resource_group'
        logger.warning("group_by was not provided but resource_group is in the dataframe, using resource_group...")
    
    # Plot for each subscription
    for subscription in anomalies_df['subscription_name'].unique():
        sub_data = grouped_data[grouped_data['subscription_name'] == subscription]
        sub_anomalies = anomalies_df[anomalies_df['subscription_name'] == subscription]
        
        if has_group_by:
            # Plot for each value in the group_by column
            for group_value in sub_anomalies[group_by].unique():
                group_data = sub_data[sub_data[group_by] == group_value]
                group_anomalies = sub_anomalies[sub_anomalies[group_by] == group_value]
                
                # Plot for each service with anomalies
                for service in group_anomalies['service'].unique():
                    __create_service_plot(
                        service_data=group_data[group_data['service'] == service],
                        service_anomalies=group_anomalies[group_anomalies['service'] == service],
                        subscription=subscription,
                        service=service,
                        timeline_folder=timeline_folder,
                        metric_name=metric_name,
                        anomalies_date=anomalies_date,
                        trend_line=trend_line,
                        has_group_by=True,
                        group_by=group_by,
                        group_value=group_value
                    )
        else:
            # Plot for each service with anomalies at subscription level
            for service in sub_anomalies['service'].unique():
                # Aggregate data at subscription and service level
                service_data = sub_data[sub_data['service'] == service].groupby(["subscription_name", "service", "start_date"]).agg({metric_name: 'sum'}).reset_index()
                service_anomalies = sub_anomalies[sub_anomalies['service'] == service].groupby(["subscription_name", "service", "start_date"]).agg({metric_name: 'sum'}).reset_index()
                
                __create_service_plot(
                    service_data=service_data,
                    service_anomalies=service_anomalies,
                    subscription=subscription,
                    service=service,
                    timeline_folder=timeline_folder,
                    metric_name=metric_name,
                    anomalies_date=anomalies_date,
                    trend_line=trend_line,
                    has_group_by=False,
                    group_by=None,
                    group_value=None
                )

def __create_service_plot(service_data: pd.DataFrame, service_anomalies: pd.DataFrame, 
                          subscription: str, service: str, timeline_folder: str, 
                          metric_name: str, anomalies_date: datetime, trend_line: bool, 
                          has_group_by: bool, group_by: str, group_value: str,
                          date_column: str = 'start_date'):
    """
    Create a timeline plot for a specific service showing anomalies.
    
    Parameters:
        service_data: DataFrame containing service data points
        service_anomalies: DataFrame containing service anomaly data points
        subscription: Subscription name
        service: Service name
        timeline_folder: Folder where plots will be saved
        metric_name: Name of the metric column
        anomalies_date: The date of anomalies to highlight in the plot
        trend_line: Whether to add a trend line
        has_group_by: Whether grouping is applied
        group_by: Column name used for grouping
        group_value: Value of the group_by column
    """
    # Skip if no data for this service
    if len(service_data) == 0:
        return
    
    # Sort anomalies by date
    service_anomalies = service_anomalies.sort_values('start_date')
    
    # Identify last anomaly in consecutive sequences
    show_annotation = []
    prev_date = None
    for idx, row in service_anomalies.iterrows():
        if prev_date is None or (row['start_date'] - prev_date).days > 1:
            # Start of new sequence
            if len(show_annotation) > 0:
                show_annotation[-1] = True
        show_annotation.append(False)
        prev_date = row['start_date']
    # Mark last anomaly in the final sequence
    if len(show_annotation) > 0:
        show_annotation[-1] = True
    
    # Create the title
    if has_group_by:
        resource_group_info = f" - {service_data['resource_group'].iloc[0]}" if 'resource_group' in service_data.columns else ""
        title = (f'{metric_name.capitalize()} Anomalies - {service_anomalies[date_column].iloc[-1].strftime("%d/%m/%Y")}\n'
                f'{subscription}{resource_group_info} - {service} \n'
                f'{group_by}: {group_value}\n')
    else:
        title = (f'{metric_name.capitalize()} Anomalies - {service_anomalies[date_column].iloc[-1].strftime("%d/%m/%Y")}\n'
                 f'{subscription} - {service} \n')
    
    plt.figure(figsize=(15, 8))
    
    # Plot normal data with improved styling (darker blue)
    plt.plot(service_data['start_date'], service_data[metric_name], 
            label=f'Normal {metric_name.capitalize()}', color='#0047AB', marker='o',
            linewidth=2, markersize=6)
    
    # Add trend line
    if trend_line:
        dates_numeric = mdates.date2num(service_data['start_date'])
        z = np.polyfit(dates_numeric, service_data[metric_name], 1)
        p = np.poly1d(z)
        plt.plot(service_data['start_date'], p(dates_numeric), 
                color='gray', linestyle='--', alpha=0.8, 
                label='Trend Line')
    
    plt.scatter(service_anomalies['start_date'], service_anomalies[metric_name],
        color='red', marker='x', s=150, label='Anomalies', zorder=5)

    if anomalies_date:
        plt.axvline(x=anomalies_date, color='r', linestyle='--', alpha=0.3)
    
    # Add annotations only for last anomaly in sequences
    for idx, (_, row) in enumerate(service_anomalies.iterrows()):
        if idx < len(show_annotation) and show_annotation[idx]:
            plt.annotate(f'€{row[metric_name]:.2f}',
                       (row['start_date'], row[metric_name]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(facecolor='white', edgecolor='red', alpha=0.7),
                       fontsize=9)
    
    plt.title(title, pad=20, fontsize=12, fontweight='bold')
    plt.xlabel('Date', fontsize=10)
    plt.ylabel(f'{metric_name.capitalize()} (€)', fontsize=10)
    
    # Improve grid and axis formatting
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add cost statistics in the corner (removed min cost)
    stats_text = (f'Mean {metric_name.capitalize()}: €{service_data[metric_name].mean():.2f}\n'
                f'Max {metric_name.capitalize()}: €{service_data[metric_name].max():.2f}')
    plt.text(0.02, 0.98, stats_text,
            transform=plt.gca().transAxes,
            bbox=dict(facecolor='white', edgecolor='gray', alpha=0.7),
            verticalalignment='top',
            fontsize=9)
    
    plt.tight_layout()
    
    # Determine save location and filename
    if has_group_by:
        sub_folder = os.path.join(timeline_folder, subscription.replace(' ', '_'))
        os.makedirs(sub_folder, exist_ok=True)
        base_filename = f"{group_value.replace(' ', '_')}_{service.replace(' ', '_')}"
        output_path = os.path.join(sub_folder, f"{base_filename}_timeline.png")
        
        # Save additional information to text file
        info_file_path = os.path.join(sub_folder, f"{base_filename}_info.txt")
        with open(info_file_path, 'w') as f:
            f.write(f"Date: {anomalies_date.strftime('%d/%m/%Y') if anomalies_date else 'N/A'}\n")
            f.write(f"Subscription Name: {subscription}\n")
            if 'resource_group' in service_data.columns and group_by != 'resource_group':
                f.write(f"Resource Group: {service_data['resource_group'].iloc[0]}\n")
            f.write(f"{group_by}: {group_value}\n")
            f.write(f"Service: {service}\n")
            f.write(f"{metric_name.capitalize()}: {service_anomalies[metric_name].iloc[0]}\n")
            f.write(f"--------------------------------\n")
            f.write(f":alert: [ANOMALY DETECTION - {service_data[group_by].iloc[0]}] [OPEN] :unlock:\n")
            f.write(f"FYI - \n\n")
            f.write(f"Description: \n")
            f.write(f"We've find an increase on the following {service} costs on {anomalies_date.strftime('%d%b')}.\n\n")
            f.write(f"Subscription Name: {subscription}\n")
            f.write(f"Resource Group: {service_anomalies['resource_group'].iloc[0]}\n")
            f.write(f"{group_by.replace('_', ' ').capitalize() + ': ' + service_anomalies[group_by].iloc[0] if group_by != 'resource_group' else ''}\n")
            f.write("   - Was there any change in the environment?\n")
            f.write("   - Is it an expected increase?\n")
            f.write("   - Is it a one-time occurrence or a new behavior from now on?")
    else:
        base_filename = f"{subscription.replace(' ', '_')}_{service.replace(' ', '_')}"
        output_path = os.path.join(timeline_folder, f"{base_filename}_timeline.png")
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate graphs for anomalies data')
    parser.add_argument('--file', type=str, help='Path to the anomalies CSV file')
    parser.add_argument('--output', type=str, help='Directory to save the generated graphs')
    args = parser.parse_args()
    
    generate_anomaly_graphs(args.file, args.output)

if __name__ == "__main__":
    main()
