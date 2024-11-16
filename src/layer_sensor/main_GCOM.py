import numpy as np
import requests
import xarray as xr
from datetime import timedelta, timezone
from shapely.geometry import box
import geopandas as gpd
import os
import matplotlib.pyplot as plt
from shapely import Polygon


from dotenv import load_dotenv
import os
import geopandas as gpd
from shapely import Polygon

# Load environment variables from the .env file
load_dotenv('/Users/hbanafsh/Documents/GitHub/Code-lab/src/a.env')

# Retrieve the path_shp variable
path_shp = os.getenv('path_shp')

# Print the output of path_shp
print(f"The value of path_shp is: {path_shp}")

# Use os.path.join to construct the full path to the shapefile
shapefile_path = os.path.join(path_shp, "WBD_10_HU2.shp")

# Print the constructed shapefile path for debugging
print(f"The full shapefile path is: {shapefile_path}")

# Read the shapefile using Geopandas
mo_basin = gpd.read_file(shapefile_path)

# Construct a geoseries with the exterior of the basin and WGS84 coordinates
mo_basin = gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")


path_snodas = os.getenv('path_snodas')
file_path = os.path.join(path_snodas, "snodas-merged.nc")
#file_path ='/Users/hbanafsh/ASU Dropbox/Hadis Banafsheh/SOS Planning/snodas-merged.nc'

# Open the dataset
ds = xr.open_dataset(file_path)

if not ds.rio.crs:
    ds = ds.rio.write_crs("EPSG:4326")

masked_ds = ds.rio.clip(mo_basin.geometry, mo_basin.crs)

# Compute SWE values from the clipped dataset
swe = masked_ds['Band1']

# Squeeze the 'band' dimension if it exists and has a size of 1
if 'band' in swe.dims and swe.sizes['band'] == 1:
    swe = swe.squeeze('band')

# Check the range of SWE values to verify non-zero values
print("SWE min:", swe.min().values, "SWE max:", swe.max().values)

# Define constants for beta2 calculation
T = 150  # Threshold for the logistic function
k = 0.03  # Scaling factor for the logistic function
epsilon = 0.05  

# Define the beta2 calculation function with an intercept
def calculate_beta2(swe_value, threshold=T, k_value=k, intercept=epsilon):
    # Logistic function with intercept (eta is bounded below by the intercept value)
    return intercept + (1 - intercept) / (1 + np.exp(k_value * (swe_value - threshold)))

# Apply the beta2 calculation to the entire SWE dataset
beta2_values = calculate_beta2(swe)

# Replace NaN values (i.e., no SWE data) with 1 (eta = 1)
beta2_values = beta2_values.fillna(1)

# Add beta2 values as a new variable in the dataset
beta2_da = xr.DataArray(
    beta2_values,
    coords={
        'time': swe['time'],  # Ensure time is included as a coordinate
        'y': swe['y'],
        'x': swe['x']
    },
    dims=['time', 'y', 'x'],  # Specify the dimensions including time
    name='beta2'
)

# Create a new dataset with beta2 values
new_ds = xr.Dataset({
    'beta2': beta2_da
})

# Remove 'grid_mapping' attribute if it exists in the dataset
for var in new_ds.variables:
    if 'grid_mapping' in new_ds[var].attrs:
        del new_ds[var].attrs['grid_mapping']

path = '/Users/hbanafsh/ASU Dropbox/Hadis Banafsheh/SOS Planning/Efficiency_files/Efficiency_resolution20_Optimization/'
file_name_preprocessed = 'Efficiency_Sensor_dataset_GCOM.nc'

new_ds.to_netcdf(path + file_name_preprocessed)

print("dataset saved to 'Efficiency_Sensor_dataset_GCOM.nc'")

