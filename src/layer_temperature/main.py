import os
from pyhdf.SD import SD, SDC
import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import earthaccess
from shapely.geometry import Point
from datetime import datetime
from dotenv import load_dotenv
import geopandas as gpd
from shapely import Polygon
import requests

# Define the start and end dates for processing
start_date = datetime(2024, 1, 20)
end_date = datetime(2024, 1, 21)
print(f"Start Date: {start_date}")
print(f"End Date: {end_date}")

# Define the date range for file processing
dates = pd.date_range(start=start_date, end=end_date, freq='D')
print(f"Dates to process: {dates}")

# Load environment variables
load_dotenv('/Users/hbanafsh/Documents/GitHub/Code-lab/src/a.env')

# Retrieve the shapefile path
path_shp = os.getenv('path_shp')
if not path_shp:
    raise ValueError("Environment variable 'path_shp' is not set.")
print(f"The value of path_shp is: {path_shp}")

shapefile_path = os.path.join(path_shp, "WBD_10_HU2.shp")

# Load the shapefile
mo_basin = gpd.read_file(shapefile_path)
mo_basin = gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")


# # Load the Missouri River Basin shapefile
# shapefile_path = os.path.join(path_shp, "WBD_10_HU2.shp")
# mo_basin = gpd.read_file(shapefile_path)
# mo_basin = gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")

# Function to convert HDF dataset to xarray DataArray
def convert_to_xarray(hdf_file, dataset_name):
    data = hdf_file.select(dataset_name)
    data_array = data[:]
    attrs = data.attributes()
    dims = [dim_name for dim_name, _ in data.dimensions().items()]
    data_array = xr.DataArray(data_array, name=dataset_name, dims=dims, attrs=attrs)
    fill_value = attrs.get('_FillValue', None)
    if fill_value is not None:
        data_array = data_array.where(data_array != fill_value, np.nan)
    return data_array

# Function to perform bilinear interpolation
def bilinear_interpolation(da):
    interpolated_slices = []
    for t in range(da.shape[0]):
        slice_data = da.isel(time=t)
        y, x = np.where(~np.isnan(slice_data.values))
        values = slice_data.values[~np.isnan(slice_data.values)]
        points = np.array(list(zip(y, x)))
        grid_y, grid_x = np.mgrid[0:slice_data.shape[0], 0:slice_data.shape[1]]
        interpolated = griddata(points, values, (grid_y, grid_x), method='linear')
        remaining_nans = np.isnan(interpolated)
        if np.any(remaining_nans):
            interpolated[remaining_nans] = griddata(points, values, (grid_y[remaining_nans], grid_x[remaining_nans]), method='nearest')
        interpolated_slices.append(
            xr.DataArray(
                interpolated,
                coords=slice_data.coords,
                dims=slice_data.dims,
                name=slice_data.name,
                attrs=slice_data.attrs,
            )
        )
    return xr.concat(interpolated_slices, dim='time')

# Function to create a mask for the dataset based on the Missouri River Basin
def create_basin_mask(lats, lons, basin_geometry):
    mask = np.zeros(lats.shape, dtype=bool)
    for i in range(lats.shape[0]):
        for j in range(lats.shape[1]):
            point = Point(lons[i, j], lats[i, j])
            if basin_geometry.contains(point):
                mask[i, j] = True
    return mask

# Function to search and download datasets from NASA Earthdata
def download_latest_datasets(collection, start_date, end_date):
    earthaccess.login(strategy="netrc")
    results = earthaccess.search_data(
        short_name=collection,
        temporal=(start_date, end_date),
    )
    local_path = os.getcwd()
    files = earthaccess.download(results, local_path=local_path)
    return files

# Define the dataset collection and download data
collection = "AIRS3STD"
file_paths = download_latest_datasets(collection, start_date, end_date)

# Check the number of files downloaded
if len(file_paths) != len(dates):
    print(f"Warning: Number of files downloaded ({len(file_paths)}) does not match the number of dates ({len(dates)}).")
available_dates = dates[:len(file_paths)]

# Process each file and store datasets
datasets = []
for i, (file_path, date) in enumerate(zip(file_paths, available_dates)):
    if os.path.exists(file_path):
        hdf_file = SD(file_path, SDC.READ)
        temp_data_array = convert_to_xarray(hdf_file, 'SurfAirTemp_A')
        latitudes = hdf_file.select('Latitude')[:]
        longitudes = hdf_file.select('Longitude')[:]
        if latitudes[0, 0] > latitudes[-1, 0]:
            print(f"Flipping latitude for file {file_path}")
            latitudes = np.flip(latitudes, axis=0)
            temp_data_array = temp_data_array.isel({'YDim:ascending': slice(None, None, -1)})
        ds = xr.Dataset({'SurfAirTemp_A': temp_data_array})
        ds = ds.assign_coords({
            'lat': (('YDim:ascending', 'XDim:ascending'), latitudes),
            'lon': (('YDim:ascending', 'XDim:ascending'), longitudes),
        })
        ds = ds.expand_dims(time=[date])
        datasets.append(ds)
    else:
        print(f"File not found: {file_path}")

# Perform bilinear interpolation and combine datasets
interpolated_datasets = [ds.assign(SurfAirTemp_A=bilinear_interpolation(ds['SurfAirTemp_A'])) for ds in datasets]
combined_dataset = xr.concat(interpolated_datasets, dim='time')

# Apply a buffered mask to the combined dataset
buffer_size = 7
mo_basin_geom_buffered = mo_basin.unary_union.buffer(buffer_size)
mask = create_basin_mask(combined_dataset['lat'].values, combined_dataset['lon'].values, mo_basin_geom_buffered)
mask_da = xr.DataArray(
    mask,
    coords={
        'YDim:ascending': combined_dataset['lat']['YDim:ascending'],
        'XDim:ascending': combined_dataset['lon']['XDim:ascending'],
    },
    dims=['YDim:ascending', 'XDim:ascending'],
)
masked_combined_dataset = combined_dataset.where(mask_da, drop=True)

# Save the masked dataset
output_path = '/Users/hbanafsh/ASU Dropbox/Hadis Banafsheh/SOS Planning/Efficiency_files/Efficiency_resolution20_Optimization/'
masked_combined_dataset.to_netcdf(os.path.join(output_path, 'Temperature_dataset.nc'))
print("Temperature dataset saved successfully.")

#Efficiency

files_to_download = [
    ("https://www.dropbox.com/scl/fi/jfgcwmav28oylggr9ezci/Temperature_dataset.nc?rlkey=fo31b5vkphei1rm4o6urupj3w&st=gvauxdo0&dl=1", "Temperature_dataset.nc"),
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


# Define constants for eta calculation
T = 0  # Reference temperature in Celsius
k = 0.5  # Increased steepness parameter
b = 0  # No horizontal shift

# Load the masked dataset
masked_combined_dataset = xr.open_dataset('Temperature_dataset.nc')
print("Time Dimension:")
print(masked_combined_dataset['time'])

# Initialize an array to hold the eta0 values
eta0_values = np.empty_like(masked_combined_dataset['SurfAirTemp_A'].values)

# Iterate over each time slice to calculate eta0
for t in range(len(masked_combined_dataset['time'])):
    # Extract temperature and convert from Kelvin to Celsius
    temp_frame = masked_combined_dataset['SurfAirTemp_A'].isel(time=t).values - 273.15
    
    # Calculate eta0 values while ignoring NaNs
    exponent = np.where(~np.isnan(temp_frame), k * (temp_frame - T) + b, np.nan)
    eta0_values[t] = np.where(~np.isnan(exponent), 1 / (1 + np.exp(exponent)), np.nan)

# Convert the eta0_values array back to an xarray.DataArray
eta0_dataarray = xr.DataArray(
    eta0_values,
    coords=masked_combined_dataset['SurfAirTemp_A'].coords,
    dims=masked_combined_dataset['SurfAirTemp_A'].dims,
    name='eta0'
)

# Create a new dataset for eta0
eta0_dataset = eta0_dataarray.to_dataset(name='eta0')

# Copy the attributes from the original dataset (optional)
eta0_dataset.attrs = masked_combined_dataset.attrs
eta0_dataset['eta0'].attrs = masked_combined_dataset['SurfAirTemp_A'].attrs

# Save the new dataset
#eta0_dataset.to_netcdf('eta0_dataset.nc')

output_path = '/Users/hbanafsh/ASU Dropbox/Hadis Banafsheh/SOS Planning/Efficiency_files/Efficiency_resolution20_Optimization/'
eta0_dataset.to_netcdf(os.path.join(output_path, 'Efficiency_Temperature_dataset.nc'))
print("Efficiency_Temperature_dataset saved successfully.")

print("eta0_dataset saved successfully.")
# Clean up temporary files
for file_path in file_paths:
    try:
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    except OSError as e:
        print(f"Error deleting file {file_path}: {e}")
