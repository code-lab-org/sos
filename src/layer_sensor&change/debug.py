import xarray as xr
import numpy as np
import geopandas as gpd
from shapely.geometry import box, Point
import gc
import requests
from dotenv import load_dotenv
import os
import gc
from shapely import Polygon
import requests


# Load environment variables from the .env file
load_dotenv('/Users/hbanafsh/Documents/GitHub/Code-lab/src/a.env')

# # Retrieve the path_shp variable
# path_shp = os.getenv('path_shp')
# print(f"The value of path_shp is: {path_shp}")

# # Define the shapefile path
# shapefile_path = os.path.join(path_shp, "WBD_10_HU2.shp")

# # Read the shapefile using Geopandas and define the Missouri River Basin
# mo_basin = gpd.read_file(shapefile_path)
# mo_basin = gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

# # Print the MO Basin CRS for verification
# print(f"MO Basin CRS: {mo_basin.crs}")


# Path_efficiency = os.getenv('Path_efficiency')
# Sensor_capella_path = os.path.join(Path_efficiency, "Efficiency_Sensor_dataset_Capella.nc")

# beta2 = xr.open_dataset('Efficiency_Sensor_dataset_Capella.nc', chunks={'time': 1, 'y': 50, 'x': 50})

# print("Variables in Efficiency_Sensor_dataset_Capella.nc:", list(beta2.variables.keys()))

# output_path = '/Users/hbanafsh/ASU Dropbox/Hadis Banafsheh/SOS Planning/Efficiency_files/Efficiency_resolution20_Optimization/'
# # Save the final coarsened dataset in the specified directory
# file_path = os.path.join(output_path, file_name_preprocessed)
# coarsened_data_masked.to_netcdf(file_path)

import xarray as xr
import os

# Define paths using environment variables and os.path.join
Path_efficiency = os.getenv('Path_efficiency')
Sensor_capella_path = os.path.join(Path_efficiency, "Efficiency_Sensor_dataset_Capella.nc")
Sensor_gcom_path = os.path.join(Path_efficiency, "Efficiency_Sensor_dataset_GCOM.nc")
SWE_change_path = os.path.join(Path_efficiency, "Efficiency_SWE_Change_dataset.nc")

# Open the datasets
try:
    sensor_capella_ds = xr.open_dataset(Sensor_capella_path, engine="netcdf4")
    print(f"Successfully opened {Sensor_capella_path}")

    sensor_gcom_ds = xr.open_dataset(Sensor_gcom_path, engine="netcdf4")
    print(f"Successfully opened {Sensor_gcom_path}")

    swe_change_ds = xr.open_dataset(SWE_change_path, engine="netcdf4")
    print(f"Successfully opened {SWE_change_path}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except ValueError as e:
    print(f"Error opening file: {e}")
