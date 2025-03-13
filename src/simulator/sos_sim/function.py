# This Script stores all the functions that are used in the simulator code
# The functions are stored in a separate file to make the main code more readable

from typing import List,Tuple

# Importing Libraries
from collections import namedtuple
import pandas as pd
import os
import geopandas as gpd
from shapely import Geometry
from datetime import datetime, timedelta, timezone
from tatc.schemas import PointedInstrument, WalkerConstellation, SunSynchronousOrbit
from tatc.analysis import collect_multi_observations
from tatc.utils import swath_width_to_field_of_regard, swath_width_to_field_of_view
from tatc.analysis import collect_multi_observations
from tatc.schemas import Satellite
from tatc.schemas import Point
from tatc.analysis import collect_ground_track

# Configure Constellation

def Snowglobe_constellation(start: datetime) -> List[Satellite]:
    roll_angle = (30 + 33.5) / 2
    roll_range = 33.5 - 30
    start = datetime(2019, 3, 1, tzinfo=timezone.utc)
    # start = datetime(2019, 3, 1, tzinfo=timezone.utc)
    constellation = WalkerConstellation(
        name="SnowGlobe Ku",
        orbit=SunSynchronousOrbit(
            altitude=555e3,
            equator_crossing_time="06:00:30",
            equator_crossing_ascending=False,
            epoch=start,
        ),
        number_planes=1,
        number_satellites=5,
        instruments=[
            PointedInstrument(
                name="SnowGlobe Ku-SAR",
                roll_angle=-roll_angle,
                field_of_regard=2 * roll_angle
                + swath_width_to_field_of_regard(555e3, 50e3),
                along_track_field_of_view=swath_width_to_field_of_view(555e3, 50e3, 0),
                cross_track_field_of_view=roll_range
                + swath_width_to_field_of_view(555e3, 50e3, roll_angle),
                is_rectangular=True,
            )
        ],
    )
    satellites = constellation.generate_members()
    # satellite_dict = {sat.name: sat for sat in satellites}
    return satellites  # , satellite_dict


# Compute next observation opportunity using TATC collect observations

from joblib import Parallel, delayed

def compute_opportunity(
    constellation: List[Satellite],
    time: datetime,
    duration: timedelta,
    requests: List[dict],
) -> gpd.GeoSeries:
    # filter requests
    filtered_requests = requests
    if filtered_requests:
        filtered_requests = [
        request
        for request in requests
        if request.get("status") is None or pd.isna(request.get("status"))
        ]
        observation_results = Parallel(-1)(
                delayed(collect_multi_observations)(
                    point, constellation, time, time + duration
                )
                for point in filtered_requests
            )
        if observation_results:
            observations = pd.concat(observation_results, ignore_index=True).sort_values(by="epoch", ascending=True)
            return observations.iloc[0]
        return None
    return None
   

# Computing Groundtrack and formatting into a dataframe

def compute_ground_track_and_format(
    sat_object: Satellite, observation_time: datetime
) -> Geometry:
    results = collect_ground_track(sat_object, [observation_time], crs="spice")
    # Formatting the dataframe
    return results.iloc[0]["geometry"]


# Code to filter requests
# def filter_requests(requests: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
#     filtered_req = requests[
#         requests["simulation_status"].isna() | (requests["simulation_status"] == "None")
#     ]
#     return filtered_req

# CALLBACK FUNCTIONS
# Reading master file

def read_master_file(date):
    # request_data = gpd.read_file("Master_file.geojson")
    print("Reading Master file")
    if os.path.exists(f"master_{date}.geojson"):     
        request_data = gpd.read_file(f"master_{date}.geojson")
        request_points = request_data.apply(
            lambda r:{
                "point":Point(id=r["simulator_id"], latitude=r["planner_latitude"], longitude=r["planner_longitude"]),
                "status":r["simulator_simulation_status"],
                "time":r["planner_time"],
                "completion_date":r["simulator_completion_date"],
                "satellite":r["simulator_satellite"],
                "polygon_groundtrack":r["simulator_polygon_groundtrack"]
            },
            axis=1
        )
    else:       
        print(f"File local_master_{date}.geojson not found. Returning an empty list.")
        request_points = []
    # request_points= request_points.to_dict('records')
    return request_points


# Update Requests in temporary dataframe
def update_requests(requests, collected_observation):
    merged = requests.merge(collected_observation, on="id", how="left")
    return merged

# Write to Master File
# Occurs at fixed time step

# def write_back_to_appender(observations_list,time):

#     # Filter the observations based on matching day/date
#     filtered_observations = [
#         observation for observation in observations_list
#         if datetime.fromtimestamp(observation['epoch_time']).date() == time.date() 
#     ]
#     return filtered_observations
    
def write_back_to_appender(source, time):
    filtered_observations = [
        observation for observation in source.requests
        if datetime.fromtimestamp(observation['epoch_time']).date() == time.date() 
    ]
    source.app.send_message("topic", "payload")
    



