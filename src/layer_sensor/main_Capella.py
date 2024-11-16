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

masked_ds = ds.rio.clip(mo_basin.geometry, mo_basin.crs)

# Compute SWE values from the clipped dataset
swe = masked_ds['Band1']

# Squeeze the 'band' dimension if it exists and has a size of 1
if 'band' in swe.dims and swe.sizes['band'] == 1:
    swe = swe.squeeze('band')

# Check the range of SWE values to verify non-zero values
print("SWE min:", swe.min().values, "SWE max:", swe.max().values)

# For Capella: Set beta2 to 1 where SWE is non-NaN, NaN otherwise
# We modify this to ensure NaN values are treated as eta = 1.
beta2_values = xr.where(np.isnan(swe), 1, xr.ones_like(swe))

# Create a DataArray for beta2 values
beta2_da = xr.DataArray(
    beta2_values,
    coords={
        'time': swe['time'], 
        'y': swe['y'],
        'x': swe['x']
    },
    dims=['time', 'y', 'x'],  
    name='beta2'
)

# Save the beta2 dataset to a NetCDF file for Capella
new_ds = xr.Dataset({
    'beta2': beta2_da
})

# Remove 'grid_mapping' attribute if it exists in the dataset
for var in new_ds.variables:
    if 'grid_mapping' in new_ds[var].attrs:
        del new_ds[var].attrs['grid_mapping']

# Save the new dataset to a NetCDF file
path = '/Users/hbanafsh/ASU Dropbox/Hadis Banafsheh/SOS Planning/Efficiency_files/Efficiency_resolution20_Optimization/'
file_name_preprocessed = 'Efficiency_Sensor_dataset_Capella.nc'

new_ds.to_netcdf(path + file_name_preprocessed)
print("dataset saved for Capella to 'Efficiency_Sensor_dataset_Capella.nc'")
