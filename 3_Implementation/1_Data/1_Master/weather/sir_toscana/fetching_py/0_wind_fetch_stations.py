import sys
import csv
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

# 1. Get the directory of the current script
current_dir = Path(__file__).resolve().parent
implementation_root = current_dir.parents[4]
if str(implementation_root) not in sys.path:
    sys.path.insert(0, str(implementation_root))

from Utils.paths import SIR_TOSCANA_PATH

# 1. Configuration
csv_filename = SIR_TOSCANA_PATH+"anemometri_stations_firenze.csv"
output_folder = SIR_TOSCANA_PATH+"sir_wind_toscana_datasets"
wind_base_url = "https://www.sir.toscana.it/archivio/download.php?IDST=anemo0_24&IDS="

# Safety delay between downloads to avoid overloading the SIR server
request_delay_seconds = 5

# Create the destination directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: '{output_folder}'")

# 2. Read station codes and trigger download requests
try:
    with open(csv_filename, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        # Verify column name exists
        if 'id_stazion' not in reader.fieldnames:
            print(f"Error: Could not find 'id_stazion' column. Available columns: {reader.fieldnames}")
            exit()
            
        stations = list(reader)
        total_stations = len(stations)
        print(f"Found {total_stations} stations to download.\n")

        for index, row in enumerate(stations, 1):
            code = row['id_stazion'].strip()
            station_name = row.get('nome', code).strip().replace(" ", "_").replace("/", "-")
            
            # Construct the target URL and local file path
            download_url = f"{wind_base_url}{code}"
            
            # Note: The SIR download handler usually exports a file (.csv) or text table
            filename_pattern = f"{code}_{station_name}.*"
            existing_files = list(Path(output_folder).glob(filename_pattern))
            if existing_files:
                print(f"[{index}/{total_stations}] Skipping {code} ({row.get('nome', '')}): already downloaded -> {existing_files[0].name}")
                continue
            
            local_filename = os.path.join(output_folder, f"{code}_{station_name}.csv")
            
            print(f"[{index}/{total_stations}] Downloading data for {code} ({row.get('nome', '')})...")
            
            try:
                # Add a User-Agent header so the server recognizes it as a standard browser request
                req = urllib.request.Request(
                    download_url, 
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
                )
                
                with urllib.request.urlopen(req, timeout=20) as response, open(local_filename, 'wb') as out_file:
                    out_file.write(response.read())
                
                print(f"    -> Saved successfully to {local_filename}")
                
            except urllib.error.HTTPError as e:
                print(f"    X HTTP Error {e.code}: Could not fetch dataset for {code}")
            except urllib.error.URLError as e:
                print(f"    X Connection Error: {e.reason}")
            except Exception as e:
                print(f"    X Unexpected error: {e}")
                
            # Crucial: safety buffer to avoid hammering the SIR server
            # and reduce the risk of temporary IP blocking.
            time.sleep(request_delay_seconds)

    print("\n All download operations completed successfully!")

except FileNotFoundError:
    print(f"Error: The file '{csv_filename}' was not found in this directory.")