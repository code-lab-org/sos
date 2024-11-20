import gurobipy as gp
from gurobipy import GRB
import geopandas as gpd
import numpy as np
import gc
import geopandas as gpd
from shapely import Polygon, box
import matplotlib.pyplot as plt
import os
from datetime import datetime, timezone, timedelta
from tatc import utils
from tatc.schemas import Instrument, Satellite, SunSynchronousOrbit, TwoLineElements
import pandas as pd
from tatc.analysis import compute_ground_track
from joblib import Parallel, delayed
from dotenv import load_dotenv
import os
import geopandas as gpd
from shapely import Polygon
import xarray as xr
import rioxarray

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


start = datetime(2024, 1, 20, tzinfo=timezone.utc)
end = datetime(2024, 1, 21, tzinfo=timezone.utc)
frame_duration = timedelta(days=1)
# compute the number of frames (time units) within the scenario
num_frames = int(1 + (end - start) / frame_duration)


# define the instrument with field of regard computed based on nominal altitude and swath width
# 700 km altitude from https://space.oscar.wmo.int/satellites/view/gcom_w
# 1450 km swath width from https://space.oscar.wmo.int/instruments/view/amsr2
amsr2 = Instrument(
    name="AMSR2",
    field_of_regard=utils.swath_width_to_field_of_regard(700e3, 1450e3),
)
# 666 km altitude from https://space.oscar.wmo.int/satellites/view/gosat_gw
# 1450 km swath width from https://space.oscar.wmo.int/instruments/view/amsr3
SAR = Instrument(
    name="SAR",
    field_of_regard=utils.swath_width_to_field_of_regard(500e3, 30e3)+30, #500 km swath at 30 km altitude
)


# define the satellite based on an orbit specification
# for existing satellites, use the two-line elements to define obrital state
# orbit from https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle
gcom_w = Satellite(
    name="GCOM-W",
    orbit=TwoLineElements(
        tle = [
            "1 38337U 12025A   24117.59466874  .00002074  00000+0  46994-3 0  9995",
            "2 38337  98.2005  58.4072 0001734  89.8752  83.0178 14.57143724635212"
        ]
    ),
    instruments=[amsr2],
)
# for future satellites, construct a new "theoretical" orbit
# orbit from https://space.oscar.wmo.int/satellites/view/gosat_gw
Capella_14 = Satellite(
    name="Capella_14",
    orbit=TwoLineElements(
        tle = [
            "1 59444U 24066C   24147.51039534  .00009788  00000+0  10218-2 0  9997",
            "2 59444  45.6083 186.3601 0001084 293.2330  66.8433 14.90000003  7233"
        ]
    ),
    instruments=[SAR],)

# compose satellite-instrument pairs to be considered for analysis
satellite_instrument_pairs = [(Capella_14, SAR)]
#satellite_instrument_pairs = [(gcom_w, amsr2)]


# define a helper function to compute the TAT-C ground tracks for a single frame
def get_ground_tracks(start, frame_duration, frame, satellite_instrument_pairs, clip_geo):
    """
    Helper function to compute all ground tracks for a given frame.

    Args:
        start (datetime): the scenario start time
        frame_duration (timedelta): the duration of each frame
        frame (int): the frame number to be computed
        satellite_instrument_pairs (list): the satellite-instrument combinations to be computed
        clip_geo (GeoSeries): the spatial domain in which to constrain results
    
    Returns:
        pd.DataFrame: a dataframe consisting of all observation polygons in this frame
    """
    return pd.concat(
        [
            compute_ground_track(
                satellite_instrument_pair[0], # satellite
                satellite_instrument_pair[1], # instrument
                pd.date_range(
                    start + frame*frame_duration, 
                    start + (frame + 1)*frame_duration,
                    freq=timedelta(seconds=10)
                ),
                crs="EPSG:3857"
            ) 
            for satellite_instrument_pair in satellite_instrument_pairs
        ]
    ).clip(clip_geo)


# parallel-process the ground track calculation for all frames
ground_tracks_Capella = pd.concat(
    Parallel(n_jobs=-1)(
        delayed(get_ground_tracks)(
            start, 
            frame_duration, 
            frame, 
            satellite_instrument_pairs,
            mo_basin.envelope
        ) 
        for frame in range(num_frames)
    ),
    ignore_index=True
)
#display(ground_tracks_Capella)
print(ground_tracks_Capella.head())

# define the satellite based on an orbit specification
# for existing satellites, use the two-line elements to define obrital state
# orbit from https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle
gcom_w = Satellite(
    name="GCOM-W",
    orbit=TwoLineElements(
        tle = [
            "1 38337U 12025A   24117.59466874  .00002074  00000+0  46994-3 0  9995",
            "2 38337  98.2005  58.4072 0001734  89.8752  83.0178 14.57143724635212"
        ]
    ),
    instruments=[amsr2],
)
# for future satellites, construct a new "theoretical" orbit
# orbit from https://space.oscar.wmo.int/satellites/view/gosat_gw
Capella_14 = Satellite(
    name="Capella_14",
    orbit=TwoLineElements(
        tle = [
            "1 59444U 24066C   24147.51039534  .00009788  00000+0  10218-2 0  9997",
            "2 59444  45.6083 186.3601 0001084 293.2330  66.8433 14.90000003  7233"
        ]
    ),
    instruments=[SAR],
)
# compose satellite-instrument pairs to be considered for analysis
#satellite_instrument_pairs = [(Capella_14, SAR)]
satellite_instrument_pairs = [(gcom_w, amsr2)]

# define a helper function to compute the TAT-C ground tracks for a single frame
def get_ground_tracks(start, frame_duration, frame, satellite_instrument_pairs, clip_geo):
    """
    Helper function to compute all ground tracks for a given frame.

    Args:
        start (datetime): the scenario start time
        frame_duration (timedelta): the duration of each frame
        frame (int): the frame number to be computed
        satellite_instrument_pairs (list): the satellite-instrument combinations to be computed
        clip_geo (GeoSeries): the spatial domain in which to constrain results
    
    Returns:
        pd.DataFrame: a dataframe consisting of all observation polygons in this frame
    """
    return pd.concat(
        [
            compute_ground_track(
                satellite_instrument_pair[0], # satellite
                satellite_instrument_pair[1], # instrument
                pd.date_range(
                    start + frame*frame_duration, 
                    start + (frame + 1)*frame_duration,
                    freq=timedelta(seconds=10)
                ),
                crs="EPSG:3857"
            ) 
            for satellite_instrument_pair in satellite_instrument_pairs
        ]
    ).clip(clip_geo)

# parallel-process the ground track calculation for all frames
ground_tracks_GCOM = pd.concat(
    Parallel(n_jobs=-1)(
        delayed(get_ground_tracks)(
            start, 
            frame_duration, 
            frame, 
            satellite_instrument_pairs,
            mo_basin.envelope
        ) 
        for frame in range(num_frames)
    ),
    ignore_index=True
)
print(ground_tracks_GCOM.head())

####################################################################################

import requests

# List of Dropbox links and corresponding output filenames
import requests

# List of Dropbox links and corresponding output filenames
files_to_download = [
    ("https://www.dropbox.com/scl/fi/00bsx7padbmmgozegdixh/coarsened_eta_output_GCOM.nc?rlkey=75tzzsaanoagf6gu83s09w4g7&st=jc5c2eba&dl=1", "coarsened_eta_output_GCOM.nc"),
    ("https://www.dropbox.com/scl/fi/i90p1hazy6ns5q74me3vh/coarsened_eta_output_Capella.nc?rlkey=hzq8coi5nu7oeasb9t1gxvif8&st=mnuh4cy3&dl=1", "coarsened_eta_output_Capella.nc"),
    ("https://www.dropbox.com/scl/fi/grz39z1epi25fu49orljw/Efficiency_Temperature_dataset.nc?rlkey=no6ph07vczazq1vv5tjv85iq9&st=rojui3tu&dl=1", "Efficiency_Temperature_dataset.nc"),
    ("https://www.dropbox.com/scl/fi/7ie9jhj5d5m00y96cad8l/efficiency_snow_cover.nc?rlkey=44vgvzwixr92cmkhycqivz8i7&st=ig665qqo&dl=1", "efficiency_snow_cover.nc"),
    ("https://www.dropbox.com/scl/fi/op0nflt34tv4pd8fahl1r/efficiency_resolution_layer.nc?rlkey=0u1qa7d9xi3atvden4qa2sxnv&st=3g69r43a&dl=1", "efficiency_resolution_layer.nc")
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



def process_satellite_data(coarsened_eta_file, eta0_resampled_file, snow_cover_file, resolution_layer_file, weights, output_filename):
    """
    Function to process satellite data by applying weighted multiplication for each dataset.
    
    Parameters:
    - coarsened_eta_file (str): Filepath for coarsened ETA dataset (specific to each satellite).
    - eta0_resampled_file (str): Filepath for resampled eta0 dataset (shared between satellites).
    - snow_cover_file (str): Filepath for snow cover dataset (shared between satellites).
    - resolution_layer_file (str): Filepath for resolution layer dataset (shared between satellites).
    - weights (dict): Dictionary containing the weights for each dataset (e.g., 'eta0', 'snow_cover', 'resolution_layer').
    - output_filename (str): The name of the output NetCDF file.

    Returns:
    - final_combined_ds (xr.Dataset): The final combined dataset.
    """
    
    # Load the coarsened eta dataset (specific for each satellite)
    coarsened_eta_ds = xr.open_dataset(coarsened_eta_file).drop_vars('spatial_ref', errors='ignore')
    
    # Ensure the CRS is set for the coarsened eta dataset
    if not coarsened_eta_ds.rio.crs:
        coarsened_eta_ds = coarsened_eta_ds.rio.write_crs("EPSG:4326")

    # Select the second time step (timestep=1)
    coarsened_eta_timestep = coarsened_eta_ds['coarsened_eta_result'].isel(time=1)

    # Scale the selected time step by the weight
    coarsened_eta_scaled = coarsened_eta_timestep * weights['coarsened_eta']

    # Load the other datasets
    eta0_resampled_ds = xr.open_dataset(eta0_resampled_file).drop_vars('spatial_ref', errors='ignore')
    snow_cover_ds = xr.open_dataset(snow_cover_file).drop_vars('spatial_ref', errors='ignore')
    resolution_layer_ds = xr.open_dataset(resolution_layer_file).drop_vars('spatial_ref', errors='ignore')

    # Ensure CRS is set for all datasets (eta0, snow cover, and resolution)
    datasets = [
        (eta0_resampled_ds, 'eta0'),
        (snow_cover_ds, 'Day_CMG_Snow_Cover'),
        (resolution_layer_ds, 'Monthly_Resolution_Abs')
    ]
    
    for dataset, var_name in datasets:
        if not dataset[var_name].rio.crs:
            dataset[var_name] = dataset[var_name].rio.write_crs("EPSG:4326")

    # Select the corresponding timestep for eta0_resampled (time=1)
    eta0_scaled = eta0_resampled_ds['eta0'].isel(time=1) * weights['eta0']

    # Extract the corresponding date from the coarsened_eta_ds dataset
    corresponding_date = coarsened_eta_ds['time'].isel(time=1).values

    # Match the date in snow cover dataset
    matching_snow_cover = snow_cover_ds.sel(time=corresponding_date, method='nearest')
    snow_cover_scaled = matching_snow_cover['Day_CMG_Snow_Cover'] * weights['snow_cover']

    # Scale the resolution layer (no need for time dimension matching here)
    resolution_layer_scaled = resolution_layer_ds['Monthly_Resolution_Abs'] * weights['resolution_layer']

    # Reproject all datasets to match the coarsened_eta_scaled dataset
    eta0_resampled = eta0_scaled.rio.reproject_match(coarsened_eta_scaled)
    snow_cover_resampled = snow_cover_scaled.rio.reproject_match(coarsened_eta_scaled)
    resolution_layer_resampled = resolution_layer_scaled.rio.reproject_match(coarsened_eta_scaled)

    # Multiply the datasets element-wise (NaNs will be preserved)
    final_eta_combined = eta0_resampled * coarsened_eta_scaled * snow_cover_resampled * resolution_layer_resampled

    # Convert the result to a new dataset
    final_combined_ds = final_eta_combined.to_dataset(name='final_eta_result')

    # Save the final combined dataset to a new NetCDF file
    path = '/Users/hbanafsh/ASU Dropbox/Hadis Banafsheh/SOS Planning/Efficiency_files/Efficiency_resolution20_Optimization/'
    final_combined_ds.to_netcdf(path + output_filename)

    print(f"Weighted multiplication complete. Final result saved to '{output_filename}'.")

    return final_combined_ds

# Filepaths shared between both satellites
eta0_resampled_file = 'Efficiency_Temperature_dataset.nc'
snow_cover_file = 'efficiency_snow_cover.nc'
resolution_layer_file = 'efficiency_resolution_layer.nc'

# Weights for each dataset (these apply to both Capella and GCOM)
weights = {
    'eta0': 0.2,  # Weight for eta0
    'coarsened_eta': 1,  # Weight for coarsened eta
    'snow_cover': 0.2,  # Weight for snow cover dataset
    'resolution_layer': 0.4  # Weight for resolution layer
}

# Processing GCOM
gcom_result = process_satellite_data(
    coarsened_eta_file='coarsened_eta_output_GCOM.nc', 
    eta0_resampled_file=eta0_resampled_file,
    snow_cover_file=snow_cover_file,
    resolution_layer_file=resolution_layer_file,
    weights=weights,
    output_filename='final_eta_snow_cover_output_GCOM.nc'
)

# Processing Capella
capella_result = process_satellite_data(
    coarsened_eta_file='coarsened_eta_output_Capella.nc', 
    eta0_resampled_file=eta0_resampled_file,
    snow_cover_file=snow_cover_file,
    resolution_layer_file=resolution_layer_file,
    weights=weights,
    output_filename='final_eta_snow_cover_output_Capella.nc'
)

##############
# List of Dropbox links and corresponding output filenames
files_to_download = [
    ("https://www.dropbox.com/scl/fi/5uyh6gkfnk6iqksnbqm21/final_eta_snow_cover_output_Capella.nc?rlkey=h3fhqv22a45xfzdjbrmetxk27&st=g1l8uehy&dl=1", "final_eta_snow_cover_output_Capella.nc"),
    ("https://www.dropbox.com/scl/fi/seha1pt3bofd58c1zyqpz/final_eta_snow_cover_output_GCOM.nc?rlkey=ehww2n9pxx6hxqoxcg28ickwh&st=gdh3ie68&dl=1", "final_eta_snow_cover_output_GCOM.nc"),

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

# Load the coarsened reward datasets for Capella and GCOM
capella_ds = xr.open_dataset('final_eta_snow_cover_output_Capella.nc')
gcom_ds = xr.open_dataset('final_eta_snow_cover_output_GCOM.nc')

# Extract Capella and GCOM reward values for January (adjust month index as needed)
capella_reward_data = capella_ds['final_eta_result'].isel(month=0).values  
gcom_reward_data = gcom_ds['final_eta_result'].isel(month=0).values 

# Get x and y coordinates (assuming both Capella and GCOM have the same coordinates)
x_coords = capella_ds['x'].values  
y_coords = capella_ds['y'].values

# Load Missouri River Basin boundary as a mask
mo_basin = gpd.read_file("WBD_10_HU2.shp")
mo_basin = gpd.GeoSeries(mo_basin.iloc[0].geometry, crs="EPSG:4326")

# Load ground tracks for Capella and GCOM
ground_tracks_Capella['time'] = pd.to_datetime(ground_tracks_Capella['time']).dt.tz_localize(None)
ground_tracks_GCOM['time'] = pd.to_datetime(ground_tracks_GCOM['time']).dt.tz_localize(None)

# Select ground tracks for the second time step (2024-01-21)
capella_tracks = ground_tracks_Capella[ground_tracks_Capella['time'] == pd.to_datetime("2024-01-21 00:00:00")]
gcom_tracks = ground_tracks_GCOM[ground_tracks_GCOM['time'] == pd.to_datetime("2024-01-21 00:00:00")]

# Ensure we have valid ground tracks for both satellites
if capella_tracks.empty or gcom_tracks.empty:
    raise ValueError("No ground tracks found for the specified time step.")

# Step 1: Filter blocks that intersect Capella and GCOM ground tracks FIRST

# Convert all grid cells into polygons for reward calculation
valid_blocks = []
n_y, n_x = capella_reward_data.shape  # Get the correct shape for looping

for i in range(n_y - 1):
    for j in range(n_x - 1):
        # Create a rectangular polygon (bounding box) for the grid cell, ensuring you don't go out of bounds
        block_geom = box(
            x_coords[j], 
            y_coords[i], 
            x_coords[min(j + 1, n_x - 1)],  # Use the minimum to avoid out-of-bounds
            y_coords[min(i + 1, n_y - 1)]   # Use the minimum to avoid out-of-bounds
        )
        
        # Extract Capella and GCOM rewards, replace NaN with 0 for GCOM
        capella_reward = capella_reward_data[i, j]
        gcom_reward = gcom_reward_data[i, j] if not np.isnan(gcom_reward_data[i, j]) else 0
        
        # Store geometry and rewards
        valid_blocks.append({
            "geometry": block_geom,
            "reward_capella": capella_reward,
            "reward_gcom": gcom_reward
        })

# Convert valid blocks list to a GeoDataFrame
valid_blocks_gdf = gpd.GeoDataFrame(valid_blocks, crs="EPSG:4326")

# Capella blocks (taskable) - intersecting with Capella ground tracks
capella_blocks = valid_blocks_gdf[valid_blocks_gdf.geometry.intersects(capella_tracks.unary_union)]

# GCOM blocks (non-taskable) - intersecting with GCOM ground tracks
gcom_blocks = valid_blocks_gdf[valid_blocks_gdf.geometry.intersects(gcom_tracks.unary_union)]

# Step 2: Calculate final rewards based on overlap
final_blocks = []
for idx, row in valid_blocks_gdf.iterrows():
    capella_reward = row['reward_capella']
    gcom_reward = row['reward_gcom']
    
    # Determine if the block intersects either Capella or GCOM ground track
    in_capella = row['geometry'].intersects(capella_tracks.unary_union)
    in_gcom = row['geometry'].intersects(gcom_tracks.unary_union)
    
    # If the block is only in Capella's ground track, it retains the Capella reward
    if in_capella and not in_gcom:
        final_reward = capella_reward
    # If the block is in both Capella and GCOM's ground track, calculate the reward as Capella - GCOM
    elif in_capella and in_gcom:
        final_reward = max(0, capella_reward - gcom_reward)
    # If the block is outside both Capella and GCOM, set the reward to 0
    else:
        final_reward = 0
    
    # Append the final reward and geometry to the new list
    final_blocks.append({
        "geometry": row['geometry'],
        "final_reward": final_reward
    })

# Convert final_blocks list to a GeoDataFrame
final_blocks_gdf = gpd.GeoDataFrame(final_blocks, crs="EPSG:4326")

# Debugging: Check sample values and range
print(final_blocks_gdf.head())
print(f"Final reward stats: \n{final_blocks_gdf['final_reward'].describe()}")

# Save this dataset for further analysis or visualization
#final_blocks_gdf.to_file('final_blocks_rewards.geojson', driver="GeoJSON")
# Path to Dropbox directory
path = '/Users/hbanafsh/ASU Dropbox/Hadis Banafsheh/SOS Planning/Efficiency_files/Efficiency_resolution20_Optimization/'
file_name_geojson = 'final_blocks_rewards.geojson'

# Save the GeoDataFrame to a GeoJSON file
final_blocks_gdf.to_file(path + file_name_geojson, driver="GeoJSON")