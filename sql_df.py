import pandas as pd
import os
import glob
from pandasql import sqldf
import argparse
from datetime import datetime

def get_current_date():
    return datetime.now()

def load_cost_data():
    """
    Load all CSV files from the costs folder and combine them into a single dataframe
    """
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    costs_dir = os.path.join(current_dir, "costs")
    
    # Check if costs directory exists
    if not os.path.exists(costs_dir):
        print(f"Error: The costs directory does not exist at {costs_dir}")
        return None
    
    # Get all CSV files in the costs directory
    csv_files = glob.glob(os.path.join(costs_dir, "*.csv"))
    
    if not csv_files:
        print("Error: No CSV files found in the costs directory")
        return None
    
    # Read and combine all CSV files
    dfs = []
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    
    if not dfs:
        print("Error: Could not read any CSV files")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Convert date column to datetime
    if 'start_date' in combined_df.columns:
        combined_df['start_date'] = pd.to_datetime(combined_df['start_date'], format='%d/%m/%Y', errors='coerce')
    
    return combined_df

def execute_sql_query(query, df):
    """
    Execute SQL query on the dataframe
    """
    try:
        # Define the dataframes available to the SQL context
        # The key is to make sure 'df' is explicitly available in locals()
        locals_dict = {'df': df}
        
        # Create a function that will use the local variables including our dataframe
        pysqldf = lambda q: sqldf(q, locals_dict)
        
        # Execute the query
        result = pysqldf(query)
        return result
    except Exception as e:
        print(f"Error executing SQL query: {e}")
        return None

def main():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 50)
    # Get current date    
    current_date = get_current_date()
    print(f"Current date: {current_date}")
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Execute SQL queries on Azure cost data')
    parser.add_argument('--query', type=str, required=True, help='SQL query to execute')
    parser.add_argument('--output', type=str, required=False, help='Output file name')
    args = parser.parse_args()
    
    # Load the data
    print("Loading cost data from all CSV files in the costs folder...")
    df = load_cost_data()
    
    if df is None:
        return
    
    print(f"Loaded {len(df)} rows of data")
    
    # Execute the query
    print(f"Executing query: {args.query}")
    result = execute_sql_query(args.query, df)
    
    if result is not None:
        print("\nQuery Results:")
        print(result)
    
    if args.output:
        result.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
        
if __name__ == "__main__":
    main()
