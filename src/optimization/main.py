import gurobipy as gp
from gurobipy import GRB
import geopandas as gpd
import numpy as np
import gc
import geopandas as gpd
from shapely import Polygon, box
import matplotlib.pyplot as plt
import rioxarray as xr
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


path_f_reward = os.getenv('path_f_reward')
file_path = os.path.join(path_f_reward, "final_blocks_rewards.geojson")
print(os.path.exists(file_path))  

# Load the final reward dataset created earlier
final_blocks_gdf = gpd.read_file(file_path)


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

# # access us states shapefile for plotting context
# us_map = gpd.read_file("https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip")
# # select contiguous states and transform to WGS84 coordinates
# conus = us_map[~us_map.STUSPS.isin(["AK", "HI", "PR"])].to_crs("EPSG:4326")

# # load the watershed boundary shapefile as a geodataframe
# mo_basin = gpd.read_file(f"WBD_10_HU2.shp")
# # construct a geoseries with the exterior of the basin and WGS84 coordinates
# mo_basin = gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")


ground_tracks_Capella['time'] = pd.to_datetime(ground_tracks_Capella['time']).dt.tz_localize(None)
ground_tracks_GCOM['time'] = pd.to_datetime(ground_tracks_GCOM['time']).dt.tz_localize(None)

# Select the ground tracks for the specific date
capella_tracks = ground_tracks_Capella[ground_tracks_Capella['time'] == pd.to_datetime("2024-01-21")]
gcom_tracks = ground_tracks_GCOM[ground_tracks_GCOM['time'] == pd.to_datetime("2024-01-21")]

# Filter blocks that intersect Capella and GCOM ground tracks
capella_blocks = final_blocks_gdf[final_blocks_gdf.geometry.intersects(capella_tracks.unary_union)]
gcom_blocks = final_blocks_gdf[final_blocks_gdf.geometry.intersects(gcom_tracks.unary_union)]

# Step 1: Ensure there are no NaN or Inf values in the final rewards
capella_blocks['final_reward'] = capella_blocks['final_reward'].replace([np.inf, -np.inf], np.nan)  # Replace Inf with NaN
capella_blocks['final_reward'] = capella_blocks['final_reward'].fillna(0)  # Replace NaN with 0

# Debugging: Check if any NaN/Inf values remain
if capella_blocks['final_reward'].isna().any() or np.isinf(capella_blocks['final_reward']).any():
    print("Warning: There are still NaN or Inf values in final rewards!")
else:
    print("No NaN or Inf values found in final rewards.")

# Step 2: Set up the Gurobi optimization model
model = gp.Model("Capella_GCOM_Block_Selection")

# Decision variables for each block in Capella's ground track (binary: 1 if selected, 0 if not)
x_Capella = model.addVars(capella_blocks.index, vtype=GRB.BINARY, name="x_Capella")

# Step 3: Define the objective function
# Use the cleaned final reward from the dataset for optimization
objective = gp.quicksum(x_Capella[b] * capella_blocks.loc[b, 'final_reward'] for b in capella_blocks.index)

# Step 4: Set the objective to maximize the sum of rewards
model.setObjective(objective, GRB.MAXIMIZE)

# Step 5: Add constraint: Capella can only observe a maximum of N locations
N = 150  # Set limit for taskable blocks
model.addConstr(gp.quicksum(x_Capella[b] for b in capella_blocks.index) <= N, "Capella_budget")

# Step 6: Set Gurobi parameters to prioritize optimality and time limit
model.setParam('MIPFocus', 1)  # Focus on finding the best solution
model.setParam('TimeLimit', 60)  # Set a time limit of 60 seconds

# Step 7: Optimize the model
model.optimize()

# Step 8: Display and save the results
if model.status == GRB.OPTIMAL:
    print("Optimal solution found:")
    
    # List to hold selected blocks
    selected_blocks = []
    
    for b in capella_blocks.index:
        if x_Capella[b].x > 0.5:
            selected_blocks.append({
                "geometry": capella_blocks.loc[b, 'geometry'],  # Geometry of the block
                "final_reward": capella_blocks.loc[b, 'final_reward']  # Final reward of the block
            })
            print(f"Capella Block {b} with final reward {capella_blocks.loc[b, 'final_reward']} is selected.")
    
    # Convert selected blocks list to GeoDataFrame
    selected_blocks_gdf = gpd.GeoDataFrame(selected_blocks, crs="EPSG:4326")

    path = '/Users/hbanafsh/ASU Dropbox/Hadis Banafsheh/SOS Planning/Efficiency_files/Efficiency_resolution20_Optimization/'
    file_name_preprocessed = 'Optimization_result.geojson'

    # Save the selected blocks to a new GeoJSON file
    selected_blocks_gdf.to_file(path + file_name_preprocessed, driver='GeoJSON')
    print("Selected Capella blocks saved as 'selected_capella_blocks.geojson'.")
else:
    print("No optimal solution found.")

# Cleanup
gc.collect()