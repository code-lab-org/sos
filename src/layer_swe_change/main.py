import xarray as xr
import numpy as np
import rioxarray

from dotenv import load_dotenv
import os
import geopandas as gpd
from shapely import Polygon

# Load environment variables from the .env file
load_dotenv('/Users/hbanafsh/Documents/GitHub/Code-lab/src/a.env')

# Retrieve the path_shp variable
path_shp = os.getenv('path_shp')


print(f"The value of path_shp is: {path_shp}")

shapefile_path = os.path.join(path_shp, "WBD_10_HU2.shp")

# print(f"The full shapefile path is: {shapefile_path}")

# Read the shapefile using Geopandas
mo_basin = gpd.read_file(shapefile_path)
mo_basin = gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")


path_snodas = os.getenv('path_snodas')
file_path = os.path.join(path_snodas, "snodas-merged.nc")
#file_path ='/Users/hbanafsh/ASU Dropbox/Hadis Banafsheh/SOS Planning/snodas-merged.nc'

# Open the dataset
ds = xr.open_dataset(file_path)

if not ds.rio.crs:
    ds = ds.rio.write_crs("EPSG:4326")
# Clip dataset to Missouri River Basin (assuming you have the mo_basin as a GeoDataFrame)
# Ensure that the mo_basin has the same CRS as the dataset
masked_ds = ds.rio.clip(mo_basin.geometry, mo_basin.crs)

# Compute SWE values from the clipped dataset
swe = masked_ds['Band1']

# Squeeze any dimensions of size 1, especially for 'band'
if 'band' in swe.dims and swe.sizes['band'] == 1:
    swe = swe.squeeze('band')

# Check the range of SWE values to verify non-zero values
print("SWE min:", swe.min().values, "SWE max:", swe.max().values)

# Mask NaN and zero values before applying the difference calculation
swe_masked = swe.where(~np.isnan(swe))

# Calculate the SWE difference between consecutive time steps, keeping NaN values intact
swe_diff_abs = swe_masked.diff(dim='time').where(~np.isnan(swe_masked.diff(dim='time')))

# Set NaN values for zero differences or areas with no changes
swe_diff_abs = abs(swe_diff_abs).where(swe_diff_abs != 0, np.nan)

# Add a zero difference for the first time step to match the length
swe_diff_abs = xr.concat([xr.zeros_like(swe.isel(time=0)), swe_diff_abs], dim='time')

# Define constants for beta5 calculation
T = 10  # Threshold for the logistic function
k = 0.2  # Scaling factor for the logistic function

# Define the beta5 calculation function
def calculate_beta5(swe_change, threshold=T, k_value=k):
    return 1 / (1 + np.exp(-k_value * (swe_change - threshold)))

# Apply the beta5 calculation to SWE changes, keeping NaN values
beta5_values = calculate_beta5(swe_diff_abs)

# Replace NaN values with 1 in beta5
beta5_values = beta5_values.fillna(1)

# Create the DataArray for beta5 values
beta5_da = xr.DataArray(
    beta5_values,
    coords={
        'time': swe['time'],
        'y': swe['y'],
        'x': swe['x']
    },
    dims=swe_diff_abs.dims,
    name='beta5'
)

# Create a new dataset with beta5 values and the absolute SWE difference
new_ds = xr.Dataset({
    'beta5': beta5_da,
    'swe_diff_abs': swe_diff_abs
})

# Transpose the dataset to ensure 'time' is the first dimension
new_ds = new_ds.transpose('time', 'y', 'x')

# Remove 'grid_mapping' attribute if it exists in the dataset
for var in new_ds.variables:
    if 'grid_mapping' in new_ds[var].attrs:
        del new_ds[var].attrs['grid_mapping']

# Save the new dataset to a NetCDF file
path = '/Users/hbanafsh/ASU Dropbox/Hadis Banafsheh/SOS Planning/Efficiency_files/Efficiency_resolution20_Optimization/'
file_name_preprocessed = 'Efficiency_Sensor_dataset.nc'

new_ds.to_netcdf(path + file_name_preprocessed)

print("dataset saved to 'Efficiency_SWE_Change_dataset.nc'")

# Check min and max of the beta5 values
#print("Beta5 min:", beta5_values.min().values, "Beta5 max:", beta5_values.max().values)
