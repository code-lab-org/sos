import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from dotenv import load_dotenv
import os
import gc
from shapely import Polygon
import requests


# Load environment variables from the .env file
load_dotenv('/Users/hbanafsh/Documents/GitHub/Code-lab/src/a.env')

# Retrieve the path_shp variable
path_shp = os.getenv('path_shp')
print(f"The value of path_shp is: {path_shp}")

# Define the shapefile path
shapefile_path = os.path.join(path_shp, "WBD_10_HU2.shp")

# Read the shapefile using Geopandas and define the Missouri River Basin
mo_basin = gpd.read_file(shapefile_path)
mo_basin = gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

# Print the MO Basin CRS for verification
print(f"MO Basin CRS: {mo_basin.crs}")

# Ensure valid geometry and CRS
assert not mo_basin.is_empty.all(), "MO Basin geometry is empty."
assert mo_basin.crs == "EPSG:4326", "MO Basin CRS must be WGS84 (EPSG:4326)."


files_to_download = [
    ("https://www.dropbox.com/scl/fi/w3a93l0v4l969pna63p2d/Efficiency_SWE_Change_dataset.nc?rlkey=rx01f8861h6p5741lg4kypqrx&st=6upzn01v&dl=1", "Efficiency_SWE_Change_dataset.nc"),
    ("https://www.dropbox.com/scl/fi/c3z2pjigoyqvpg2oo52u2/Efficiency_Sensor_dataset_GCOM.nc?rlkey=78fge5ai1nixa19qtrczcdvg0&st=wy3dvwq6&dl=1", "Efficiency_Sensor_dataset_GCOM.nc"),
    ("https://www.dropbox.com/scl/fi/kak8qcsw54f3ol4se458h/Efficiency_Sensor_dataset_Capella.nc?rlkey=pjhaoe8re5zbyp7bq6nmgr3md&st=wwc0f4ia&dl=1", "Efficiency_Sensor_dataset_Capella.nc"),
]

for url, output_file in files_to_download:
    try:
        # Send a GET request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check for HTTP errors

        # Write the content to a file in chunks to avoid memory issues
        with open(output_file, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Download completed successfully! File saved as: {output_file}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred for {output_file}: {http_err}")
    except Exception as err:
        print(f"An error occurred for {output_file}: {err}")



# Load the beta datasets

beta5 = xr.open_dataset('Efficiency_SWE_Change_dataset.nc', chunks={'time': 1, 'y': 50, 'x': 50})

print("Variables in Efficiency_SWE_Change_dataset.nc:", list(beta5.variables.keys()))

# Function to process eta datasets
def process_eta_datasets(sensor_file, swe_ds, file_name_preprocessed, coarsen_factor_y=20, coarsen_factor_x=20, weight1=0.5, weight2=0.3):
    # Load the sensor dataset
    sensor_ds = xr.open_dataset(sensor_file, chunks={'time': 1, 'y': 50, 'x': 50})
    print(f"Variables in {sensor_file}:", list(sensor_ds.variables.keys()))
    
    # Process all time steps and calculate the weighted eta_result
    eta_results = []
    for time_index in range(sensor_ds.sizes['time']):
        # Extract data for the current time step
        sensor_slice = sensor_ds.isel(time=time_index)
        swe_slice = swe_ds.isel(time=time_index)
        
        # Create a valid mask for non-NaN values
        valid_mask = (~np.isnan(sensor_slice['beta2'])) & (~np.isnan(swe_slice['beta5']))
        sensor_valid = sensor_slice['beta2'].where(valid_mask)
        swe_valid = swe_slice['beta5'].where(valid_mask)
        
        # Compute weighted eta results
        eta_step1 = sensor_valid * weight1
        eta_step2 = eta_step1 * (swe_valid * weight2)
        
        # Expand time dimension and append to results
        eta_step2 = eta_step2.expand_dims(time=[sensor_slice['time'].values])
        eta_results.append(eta_step2)
        
        # Cleanup
        del sensor_slice, swe_slice, sensor_valid, swe_valid, eta_step1
        gc.collect()
    
    # Concatenate results across all time steps
    final_eta_result = xr.concat(eta_results, dim='time')
    final_eta_result['time'] = sensor_ds['time']
    final_eta_ds = final_eta_result.to_dataset(name='eta_result')
    
    # Coarsen the dataset
    coarsened_data = final_eta_ds['eta_result'].coarsen(
        y=coarsen_factor_y, x=coarsen_factor_x, boundary='trim'
    ).mean(skipna=True)
    
    # Create a mask for the coarsened grid based on whether the block's center is inside the basin
    coarsened_x_coords = coarsened_data['x'].values
    coarsened_y_coords = coarsened_data['y'].values
    valid_mask = np.zeros((len(coarsened_y_coords), len(coarsened_x_coords)), dtype=bool)
    
    for i, y in enumerate(coarsened_y_coords):
        for j, x in enumerate(coarsened_x_coords):
            block_center = Point(x, y)
            if mo_basin.contains(block_center).any():
                valid_mask[i, j] = True
    
    # Apply the valid mask to the coarsened data
    coarsened_data_masked = coarsened_data.where(valid_mask)
    #coarsened_ds = coarsened_data_masked.to_dataset(name='coarsened_eta_result')
    # Define output path
    output_path = '/Users/hbanafsh/ASU Dropbox/Hadis Banafsheh/SOS Planning/Efficiency_files/Efficiency_resolution20_Optimization/'
    # Save the final coarsened dataset in the specified directory
    coarsened_data_masked.to_netcdf(output_path + file_name_preprocessed)
    #print(f"Coarsened dataset saved as '{output_file}'.")
    
    # Cleanup
    sensor_ds.close()
    gc.collect()


# Example usage of process_eta_datasets
swe_ds = xr.open_dataset('Efficiency_SWE_Change_dataset.nc', chunks={'time': 1, 'y': 50, 'x': 50})

process_eta_datasets(
    sensor_file='Efficiency_Sensor_dataset_GCOM.nc',
    swe_ds=swe_ds,
    file_name_preprocessed = 'coarsened_eta_output_GCOM.nc'
)

process_eta_datasets(
    sensor_file='Efficiency_Sensor_dataset_Capella.nc',
    swe_ds=swe_ds,
    file_name_preprocessed = 'coarsened_eta_output_Capella.nc'
)

swe_ds.close()
gc.collect()

print("Processing complete for GCOM and Capella datasets.")
