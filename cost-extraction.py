import sys
import time
import requests
import pandas as pd
import os
from dotenv import load_dotenv
from logger import setup_logger
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from azure.identity import AzureCliCredential

# Creating Logger
logger = setup_logger('Cost Extraction')

# Authentication
api_version = "2024-08-01"
base_url = "https://management.azure.com"
br_date_format = "%d/%m/%Y"
azure_date_format = "%Y%m%d"
sleep_time = 10 # Sleep to avoid 429 errors (Too many requests)
TOKEN_HEADER = ''
RETRY_DEFAULT = 60
MONTHS_BACK = 4

# Granularity level constants
GRANULARITY_RESOURCE = 'resource'
GRANULARITY_RESOURCE_GROUP = 'resource_group'
GRANULARITY_SERVICE = 'service'  # New constant for service-level granularity

def extract_resource_info(resource_id: str):
    """Extract resource group name and resource name from ResourceID."""
    try:
        parts = resource_id.lower().split('/')
        resource_group = next((parts[i+1] for i, part in enumerate(parts) if part == 'resourcegroups'), None)
        resource_name = parts[-1] if len(parts) > 0 else None
        resource_type = parts[-2] if len(parts) > 1 else None
        return resource_group, resource_name, resource_type
    except Exception:
        return None, None, None

def get_auth_header(tenant_id: str = None):
    """Get authentication header for Azure REST API."""
    try:
        global TOKEN_HEADER
        credential = AzureCliCredential(tenant_id=tenant_id)
        logger.info("Getting authentication token...")
        token = credential.get_token(f"{base_url}/.default").token
        TOKEN_HEADER = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    except Exception as e:
        logger.error(f"Authentication error occurred. Please check your credentials.")
        sys.exit(1)


def get_cost_by_date(start: datetime, end: datetime, scope: str, tenant_id: str = None, main_dimension: str = 'ServiceName', 
                     max_retries: int = 5, subscription_name: str = None, service_name: str = None,
                     granularity_level: str = GRANULARITY_SERVICE, granularity_type: str = 'daily'):
    """
    Fetch cost data from Azure Cost Management using REST API with pagination and retry logic for 429 errors.
    """
    url = f"{base_url}/providers/{scope}/providers/Microsoft.CostManagement/query?api-version={api_version}"
    payload = {
        "type": "AmortizedCost",
        "timeframe": "Custom",
        "timePeriod": {"from": start.strftime('%Y-%m-%dT%H:%M:%SZ'), "to": end.strftime('%Y-%m-%dT%H:%M:%SZ')},
        "dataset": {
            "granularity": granularity_type,
            "aggregation": {"totalCost": {"name": "Cost", "function": "Sum"}},
            "grouping": [
                {"type": "Dimension", "name": main_dimension},
                {"type": "Dimension", "name": "SubscriptionName"}
            ]
        }
    }

    # Build filter conditions dynamically
    filter_conditions = []

    if subscription_name:
        filter_conditions.append({
            "dimensions": {
                "name": "SubscriptionName",
                "operator": "In",
                "values": [subscription_name]
            }
        })

    if service_name:
        filter_conditions.append({
            "dimensions": {
                "name": "ServiceName",
                "operator": "In",
                "values": [service_name]
            }
        })

    # Only add the filter if there are conditions
    if filter_conditions:
        payload["dataset"]["filter"] = {
            "and": filter_conditions
        }
    
    # Add appropriate grouping based on granularity level
    if granularity_level == GRANULARITY_RESOURCE:
        payload["dataset"]['grouping'].append({"type": "Dimension", "name": "ResourceId"})
    elif granularity_level == GRANULARITY_RESOURCE_GROUP:
        payload["dataset"]['grouping'].append({"type": "Dimension", "name": "ResourceGroupName"})
    # For service level, we don't need additional grouping since ServiceName and SubscriptionName 
    # are already included by default
    
    # Making requests and handling with next_links
    try:
        all_rows = []
        retries = 0
        while url:
            logger.info("Making requests...")
            response = requests.post(url, headers=TOKEN_HEADER, json=payload)

            # Check for 401 error Permission Denied
            if response.status_code == 401:
                logger.warning(f"Token expired...")
                get_auth_header(tenant_id=tenant_id)
                continue # Try again after getting a new token

            # Check for 429 error (Too Many Requests)
            if response.status_code == 429:
                retries += 1
                if retries > max_retries:
                    logger.error("Max retries reached. Aborting.")
                    break
                
                retry_after = response.headers.get('x-ms-ratelimit-microsoft.costmanagement-clienttype-retry-after', RETRY_DEFAULT)  # Default to 60 seconds if not specified
                logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
                time.sleep(int(retry_after))
                continue  # Try again after waiting

            response.raise_for_status()  # Raise error for other types of failed responses
            
            data = response.json()
            if "properties" in data and "rows" in data["properties"]:
                all_rows.extend(data["properties"]["rows"])
            url = data.get("properties", {}).get("nextLink")
            time.sleep(sleep_time)

        logger.info("Done!!")
        return all_rows

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        sys.exit(1)

def collect_cost_data(scope: str, tenant_id: str = None, subscription_name: str = None, service_name: str = None, 
                      granularity_level: str = GRANULARITY_SERVICE, months_back: int = MONTHS_BACK, granularity_type: str = 'daily'):
    """
    Collect cost data for the specified number of months back.
    """
    costs_list = []
    end = datetime.today().astimezone() - timedelta(days=1)
    start = end - relativedelta(months=months_back, day=1)
    
    rows = get_cost_by_date(
        start, end, scope, tenant_id,
        subscription_name=subscription_name, 
        service_name=service_name, 
        granularity_level=granularity_level,
        granularity_type=granularity_type
    )
    
    for item in rows:
        cost = {
            'start_date': datetime.strptime(str(item[1]), azure_date_format).strftime(br_date_format) if granularity_type == 'daily' else datetime.fromisoformat(str(item[1])).strftime(br_date_format),
            'service': item[2],
            'subscription_name': item[3],
            'cost': item[0],
            'currency': item[-1]
        }

        if granularity_level == GRANULARITY_RESOURCE:
            resource_id_value = item[4]
            resource_group, resource_name, resource_type = extract_resource_info(resource_id_value)
            cost.update({
                'resource_group': resource_group,
                'resource_name': resource_name,
            })
        elif granularity_level == GRANULARITY_RESOURCE_GROUP:
            cost['resource_group'] = item[4]
        
        costs_list.append(cost)
    return costs_list

def save_results_to_file(data: list, filename: str):
    logger.info(f"Saving results...")
    df = pd.DataFrame(data)
    df.to_csv(filename, sep=',', index=False)
    logger.info(f"Done!! check the {filename} file")

def main(scope: str = '', tenant_id: str = '', mg: str = '', months_back: int = 4, granularity_type: str = 'daily'):
    """
    Main function to collect cost data and detect anomalies.
    The first request is always at service level, subsequent detail requests use the specified granularity.
    """

    # Adding a suffix to the mg name if the granularity type is monthly
    if granularity_type == 'monthly':
        mg = mg + '_monthly'

    logger.info(f"-------------------- Starting {mg} --------------------")
    filename = f'{mg}_results.csv'

    get_auth_header(tenant_id=tenant_id)
    # First get costs at service level (less detailed, faster) 
    # _granularity_level = GRANULARITY_SERVICE if granularity_type == 'daily' else detail_granularity_level # If monthly, use the requested granularity level
    _granularity_level = GRANULARITY_SERVICE
    logger.info(f"---------- Getting initial costs at {_granularity_level} level ----------")
    costs_list = collect_cost_data(scope,
                                    granularity_level=_granularity_level,
                                    granularity_type=granularity_type,
                                    months_back=months_back)
    save_results_to_file(costs_list, filename)

if __name__ == "__main__":
    # load the environment variables
    load_dotenv()
    tenant_id = os.getenv('TENANT_ID')
    mg = os.getenv('MG')
    scope = os.getenv('SCOPE')
    if not tenant_id or not mg or not scope:
        logger.error("Error with the environment variables: Please check if the variables are set correctly in the .env file")
        sys.exit(1)
    main(scope, tenant_id, mg)