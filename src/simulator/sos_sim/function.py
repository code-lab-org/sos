# This Script stores all the functions that are used in the simulator code
# The functions are stored in a separate file to make the main code more readable

import logging
import numpy as np
import os
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.framelib import itrs
import sys
import time as t
from datetime import datetime, timedelta, timezone
from typing import List
import geopandas as gpd
import pandas as pd
from boto3.s3.transfer import TransferConfig
from joblib import Parallel, delayed
from shapely import Geometry, wkt
from tatc.analysis import collect_ground_track, collect_multi_observations
from tatc.schemas import (
    Point,
    PointedInstrument,
    Satellite,
    SunSynchronousOrbit,
    WalkerConstellation,
)
from tatc.utils import swath_width_to_field_of_regard, swath_width_to_field_of_view
logger = logging.getLogger(__name__)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.sos_tools.aws_utils import AWSUtils
from src.sos_tools.data_utils import DataUtils
from constellation_config_files.schemas import VectorLayer


def Snowglobe_constellation(start: datetime) -> List[Satellite]:
    """
    Create and configure the constellation.
    Args:
        start (datetime): The start time for the constellation.
    Returns:
        List[Satellite]: A list of Satellite objects representing the SnowGlobe constellation.
    """
    roll_angle = (30 + 33.5) / 2
    roll_range = 33.5 - 30
    start = datetime(2019, 3, 1, tzinfo=timezone.utc)
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
    return satellites

# def compute_opportunity(
#     const: List[Satellite],
#     time: datetime,
#     duration: timedelta,
#     requests: List[dict],
#     parallel_compute: bool = True,
# ) -> gpd.GeoSeries:
#     """
#     Compute the next observation opportunity for a given time and duration using TATC collect observations.
#     Args:
#         const (List[Satellite]): List of satellites in the constellation.
#         time (datetime): The time at which to compute the observation opportunity.
#         duration (timedelta): The duration for which to compute the observation opportunity.
#         requests (List[dict]): List of requests to filter.
#         parallel_compute (bool): Whether to use parallel computation for observation collection.
#     Returns:
#         gpd.GeoSeries: The observation opportunity as a GeoSeries.
#     """
#     # start_time = t.time()
#     filtered_requests = requests
#     time = time.replace(tzinfo=timezone.utc)
#     end = (time + duration).replace(tzinfo=timezone.utc)

#     filtered_requests = [
#         request
#         for request in requests
#         if request.get("simulator_simulation_status") is None
#         or pd.isna(request.get("simulator_simulation_status"))
#     ]

#     if filtered_requests:
#         # column_names = list(filtered_requests[0].keys())

#         # def collect_observations_for_request(request):
#         #     try:
#         #         return collect_multi_observations(request["point"], const, time, end)
#         #     except Exception as e:
#         #         print(f"Error processing request {request}: {e}")
#         #         return pd.DataFrame()

#         # observation_results_list = Parallel(n_jobs=-1 if parallel_compute else 1)(
#         #     delayed(collect_observations_for_request)(request)
#         #     for request in filtered_requests
#         # )

#         # # Remove any empty results
#         # observation_results_list = [df for df in observation_results_list if not df.empty]

#         # if observation_results_list:
#         #     observation_results = pd.concat(
#         #         observation_results_list, ignore_index=True
#         #     ).sort_values(by="epoch", ascending=True)
#         # else:
#         #     observation_results = pd.DataFrame()

#         observation_results = pd.concat(
#             [
#                 collect_multi_observations(request["point"], const, time, end)
#                 for request in filtered_requests
#             ],
#             ignore_index=True,
#         ).sort_values(by="epoch", ascending=True)

#         if observation_results is not None and not observation_results.empty:
#             # logger.info(f"Observation opportunity exist{time + duration}")
#             id_to_eta = {
#             request["point"].id: request["planner_final_eta"]
#             for request in filtered_requests
#             }            
#             observation_results["planner_final_eta"] = observation_results["point_id"].map(id_to_eta)
#             # observations = pd.concat(observation_results, ignore_index=True).sort_values(by="epoch", ascending=True)
#             return observation_results
#         return None
#     else:
#         return None  

# def filter_and_sort_observations(df, sim_time,incomplete_ids,time_step_constr):
    
#     # logger.info(
#     #     f"Filtering and sorting observations, type of df is {type(df)}, length of df is {len(df)}"
#     # )
#     # Filter for observations with incomplete simulation status
#     df = df[df["point_id"].isin(incomplete_ids)]
#     # Ensure sim_time is timezone-aware and same tz as df
#     if df['epoch'].dt.tz is not None and sim_time.tzinfo is None:
#         sim_time = sim_time.replace(tzinfo=df['epoch'].dt.tz)

#     # Step 1: Filter for observations within 1 minute of simulation time
#     time_step_later = sim_time + time_step_constr
#     logger.info(
#         f"Filtering observations, sim_time: {sim_time}, time_step_later: {time_step_later}, type of sim_time is {type(sim_time)}, type of time_step_later is {type(time_step_later)}")
#     mask = (df["epoch"] >= sim_time) & (df["epoch"] <= time_step_later)
#     filtered = df[mask]
    
#     logger.info(
#         f"Filtered observations, type of filtered is {type(filtered)}, length of filtered is {len(filtered)}")
    
#     # Step 2: Sort by planner_final_eta descending
#     sorted_filtered = filtered.sort_values(by="planner_final_eta", ascending=False)
#     logger.info(
#         f"Sorted observations, type of sorted is {sorted_filtered}, length of sorted is {len(sorted_filtered)}"
#     )
#     return sorted_filtered.iloc[0] if not sorted_filtered.empty else None


# Compute Opportunity 3 days
#########################################################################################################

def compute_opportunity(
    const: List[Satellite],
    time: datetime,
    duration: timedelta,
    requests: List[dict],
    parallel_compute: bool = True,
) -> gpd.GeoSeries:
    """
    Compute the next observation opportunity for a given time and duration using TATC collect observations.
    Args:
        const (List[Satellite]): List of satellites in the constellation.
        time (datetime): The time at which to compute the observation opportunity.
        duration (timedelta): The duration for which to compute the observation opportunity.
        requests (List[dict]): List of requests to filter.
        parallel_compute (bool): Whether to use parallel computation for observation collection.
    Returns:
        gpd.GeoSeries: The observation opportunity as a GeoSeries.
    """
    # start_time = t.time()
    filtered_requests = requests
    time = time.replace(tzinfo=timezone.utc)
    end = (time + duration).replace(tzinfo=timezone.utc)

    filtered_requests = [
        request
        for request in requests
        if request.get("simulator_simulation_status") is None
        or pd.isna(request.get("simulator_simulation_status"))
    ]

    if filtered_requests:       
        observation_results = pd.concat(
            [
                collect_multi_observations(request["point"], const, time, end)
                for request in filtered_requests
            ],
            ignore_index=True,
        ).sort_values(by="epoch", ascending=True)

        if observation_results is not None and not observation_results.empty:
            # logger.info(f"Observation opportunity exist{time + duration}")
            # Calulating computation time
            start = t.time()
            id_to_meta = {
                request["point"].id: {
                    "planner_final_eta": request["planner_final_eta"],
                    "collected_within_last_3_days": request["collected_within_last_3_days"]
                }
                for request in filtered_requests
            }        

            observation_results["planner_final_eta"] = observation_results["point_id"].map(
                lambda pid: id_to_meta.get(pid, {}).get("planner_final_eta")
            )

            observation_results["collected_within_last_3_days"] = observation_results["point_id"].map(
                lambda pid: id_to_meta.get(pid, {}).get("collected_within_last_3_days", False)
            )

            end_time = t.time()
            computation_time = end_time - start
            # logger.info(
            #     f"Observation opportunity computation time: {computation_time:.2f} seconds"
            # )
            # logger.info(f"observation results {observation_results}")

            return observation_results

        # return None
    else:
        # logger.info("No filtered requests found for observation opportunity.")
        return None
    

# Filter and format the observation results

def filter_and_sort_observations(df, sim_time,incomplete_ids,time_step_constr):

    # logger.info(
    #     f"Filtering and sorting observations"
    # )
    # Filter for observations with incomplete simulation status
    df = df[df["point_id"].isin(incomplete_ids)]
    # Ensure sim_time is timezone-aware and same tz as df
    if df['epoch'].dt.tz is not None and sim_time.tzinfo is None:
        sim_time = sim_time.replace(tzinfo=df['epoch'].dt.tz)

    # Step 1: Filter for observations within 1 minute of simulation time
    time_step_later = sim_time + time_step_constr

    mask = (df["epoch"] >= sim_time) & (df["epoch"] <= time_step_later)
    filtered = df[mask]   

    start_time = t.time()
    sorted_filtered = filtered.sort_values(by=["collected_within_last_3_days", "planner_final_eta"], ascending=[True, False])
    # logger.info(
    #     f"Sorted observations, type of sorted is {sorted_filtered}, length of sorted is {len(sorted_filtered)}"
    # )
    # logger.info(f"Length of filtered observation {len(sorted_filtered)}")
    end_time = t.time()
    computation_time = end_time - start_time
    # logger.info(
    #     f"Filtering and sorting observations time: {computation_time:.2f} seconds"
    # )
    
    return sorted_filtered.iloc[0] if not sorted_filtered.empty else None

######################################Filter for 3 days ends here################################################
#################################################################################################################

# Computing Groundtrack and formatting into a dataframe

def compute_ground_track_and_format(
    sat_object: Satellite, observation_time: datetime
) -> Geometry:
    """
    Compute the ground track for a given satellite object and observation time into a dataframe.
    Args:
        sat_object (Satellite): The satellite object for which to compute the ground track.
        observation_time (datetime): The time at which to compute the ground track.
    Returns:
        Geometry: The ground track as a Shapely geometry object.
    """
    # logger.info(
    #     f"Computing ground track for {sat_object.name} at {observation_time}, type of observation time is {type(observation_time)}"
    # )
    observation_time = observation_time.replace(tzinfo=timezone.utc)
    results = collect_ground_track(sat_object, [observation_time], crs="spice")
    # logger.info(f"Length of results: {len(results)}, Type of results: {type(results)}")
    return results["geometry"].iloc[0]


def read_master_file():
    """
    Read the master file and convert it to a list of dictionaries.
    Returns:
        List[dict]: A list of dictionaries representing the master file data.
    """
    # logger.info("Reading master file")
    # start_time = t.time()
    output_filename = "outputs/master.geojson"
    if os.path.exists(output_filename):
        request_data = gpd.read_file(output_filename)
        request_points = request_data.apply(
            lambda r: {
                "point": Point(
                    id=r["simulator_id"],
                    latitude=r["planner_latitude"],
                    longitude=r["planner_longitude"],
                ),
                "simulator_id": r["simulator_id"],
                "planner_time": r["planner_time"],
                "planner_latitude": r["planner_latitude"],
                "planner_longitude": r["planner_longitude"],
                "simulator_simulation_status": r["simulator_simulation_status"],
                "simulator_completion_date": r["simulator_completion_date"],
                "simulator_satellite": r["simulator_satellite"],
                "collected_within_last_3_days": r["collected_within_last_3_days"],
                "planner_final_eta": r["planner_final_eta"],
                "simulator_polygon_groundtrack": r["simulator_polygon_groundtrack"],
                "planner_geometry": r["geometry"]                
            },
            axis=1,
        ).tolist()
    else:
        # print(f"Master file not found. Returning an empty list.")
        request_points = []

    # end_time = t.time()
    # Calculate the total time taken
    # computation_time = end_time - start_time
    # logger.debug(f"Reading master file time: {computation_time:.2f} seconds")
    return request_points

# this function is triggered by scenario time interval callback, it writes the daily output to a geojson file

def write_back_to_appender(source, time):
    """
    Write the processed data back to the appender and upload it to S3.
    Args:
        source: The source object containing the data to be processed.
        time (datetime): The time at which to write back the data.
    """
    # Establish connection to S3
    s3 = AWSUtils().client
    output_directory = os.path.join("outputs", source.app.app_name)
    data_utils = DataUtils()
    data_utils.create_directories([output_directory])
    logger.info("Entering write_back_to_appender function") 
    appender_data = process_master_file(source.requests)
    selected_json_data = pd.DataFrame(appender_data)
    logger.info(f"length of data in master file {len(appender_data)}")

    # logger.info(
    #     f"Type of simulator_polygon_groundtrack{type(selected_json_data['simulator_polygon_groundtrack'])}"
    # )

    selected_json_data["simulator_polygon_groundtrack"] = selected_json_data[
        "simulator_polygon_groundtrack"
    ].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
    gdf = gpd.GeoDataFrame(selected_json_data, geometry="simulator_polygon_groundtrack")
   
    # if os.path.exists(output_filename):
    #     # computation time 
    #     start_time = t.time()
    #     logger.info(f"Entering function to update simulator columns in master file")
    #     # Read the existing master file
    #     master_gdf = gpd.read_file(output_filename)
    #     master_gdf = master_gdf.set_index("simulator_id")
    #     gdf_temp = gdf.set_index("simulator_id")
    #     # Select only columns to update from gdf
    #     cols_to_update = [
    #         "simulator_simulation_status",
    #         "simulator_completion_date",
    #         "simulator_satellite",
    #         "collected_within_last_3_days",
    #         "planner_final_eta",
    #         "simulator_polygon_groundtrack",
    #         "planner_geometry"
    #     ]
    #     # Filter gdf columns to just those
    #     gdf_subset = gdf_temp[cols_to_update]
    #     # Update master_gdf with gdf_subset values where indices match
    #     master_gdf.update(gdf_subset)
    #     # Reset index if needed
    #     master_gdf = master_gdf.reset_index()
    #     # gdf_temp = gdf.reset_index()        # Save updated master back to file
    #     master_gdf.to_file(output_filename, driver="GeoJSON")
    #     logger.info(f"Updated master file with new simulator data")
    #     end_time = t.time()
    #     computation_time = end_time - start_time
    #     logger.info(
    #         f"Time taken to update master file with new simulator data: {computation_time:.2f} seconds"
    #     )
    # gdf.to_file(f"outputs/master.geojson", driver="GeoJSON")
    # logger.info(f"{source.app.app_name} sending message.")
    date_sim_time = source.app.simulator._time
    date_sim = str(date_sim_time.date()).replace("-", "")
    # Saving Daily local files for LIS ingestion
    gdf["simulator_completion_date"] = pd.to_datetime(
        gdf["simulator_completion_date"], errors="coerce"
    )
    daily_gdf_filtered = gdf[
        gdf["simulator_completion_date"].dt.date == source.app.simulator._time.date()
    ]
    current_simulation_date = os.path.join(output_directory, str(date_sim_time.date()))
    data_utils.create_directories([current_simulation_date])
    output_file = os.path.join(
        current_simulation_date, f"simulator_output_{date_sim}.geojson"
    )
    daily_gdf_filtered.to_file(output_file)
    s3.upload_file(
        Bucket="snow-observing-systems",
        Key=output_file,
        Filename=output_file,
        Config=TransferConfig(use_threads=False),
    )
    # logger.info(f"Length of daily simulator file  : {len(daily_gdf_filtered)}")

    # Sending message to the appender
    logger.info(f"Data type daily simulator file dataframe all columns {daily_gdf_filtered.dtypes}")
    daily_gdf_filtered["simulator_completion_date"] = daily_gdf_filtered["simulator_completion_date"].astype(str)
    # daily_gdf_filtered["planner_time"] = daily_gdf_filtered["planner_time"].astype(str)
    # daily_gdf_filtered = daily_gdf_filtered.drop(columns=["point","planner_geometry","planner_time","planner"])
    
    daily_gdf_filtered = daily_gdf_filtered[
        [
            "simulator_id",
            "simulator_simulation_status",
            "simulator_completion_date",
            "simulator_satellite",
            "simulator_polygon_groundtrack"
        ]
    ]


    selected_json_data = daily_gdf_filtered.to_json()
    source.app.send_message(
        source.app.app_name,
        "simulator_daily",  # ["master", "selected"],
        VectorLayer(vector_layer=selected_json_data).model_dump_json(),

    )

    logger.info("Simulator sent message to appender")


    # Saving to master file
    # Reading master.geojson and populating the simulator columns
    # master_path = "outputs/master.geojson"
    # gdf.to_file(master_path, driver="GeoJSON")

    


def process_master_file(existing_request):
    """
    Process the master file and update the existing requests with the new data.
    Args:
        existing_request (List[dict]): The existing requests to be updated.
    Returns:
        List[dict]: The updated list of requests.
    """
    logger.info(f"Processing master file")
    # start_time = t.time()
    master = read_master_file()
    master_processed = [
        request
        for request in master
        if request["simulator_simulation_status"] == "Completed"
    ]
    master_unprocessed = [
        request for request in master if request["simulator_simulation_status"] is None
    ]
    # Code to update master_processed exisitng request based on id and if status is completed in exisitng request
    for unprocessed_request in master_unprocessed:
        for request in existing_request:
            if request["point"] == unprocessed_request["point"]:
                for key, value in request.items():
                    if  key != 'collected_within_last_3_days':
                    #     unprocessed_request["simulator_id"] = value.id
                    #     unprocessed_request["planner_latitude"] = value.latitude
                    #     unprocessed_request["planner_longitude"] = value.longitude
                    # else:
                        unprocessed_request[key] = value
                #   unprocessed_request.update(request)  # Update only fields, don't replace dict

    # end_time = t.time()
    # Calculate the total time taken
    # computation_time = end_time - start_time
    # logger.info(f"Process master file time: {computation_time:.2f} seconds")

    # Return the combined list of processed and unprocessed requests
    return master_processed + master_unprocessed


def convert_to_vector_layer_format(visual_requests):
    """
    Convert the visual requests to a GeoDataFrame and then to a GeoJSON format.
    Args:
        visual_requests (List[dict]): The list of visual requests to be converted.
    Returns:
        str: The GeoJSON representation of the visual requests.
    """
    # start_time = t.time()
    vector_data = pd.DataFrame(visual_requests)
    vector_data["geometry"] = vector_data["planner_geometry"].apply(
        lambda x: wkt.loads(x) if isinstance(x, str) else x
    )
    vector_data_gdf = gpd.GeoDataFrame(vector_data, geometry="geometry")
    vector_data_gdf["simulator_completion_date"] = vector_data_gdf[
        "simulator_completion_date"
    ].astype(str)
    vector_data_gdf["point"] = vector_data_gdf["point"].astype(str)
    vector_data_gdf["planner_time"] = vector_data_gdf["planner_time"].astype(str)
    vector_data_gdf["simulator_polygon_groundtrack"] = vector_data_gdf[
        "simulator_polygon_groundtrack"
    ].astype(str)
    vector_data_gdf["planner_geometry"] = vector_data_gdf["planner_geometry"].astype(
        str
    )
    # logger.info(f"type of vector data gdf{vector_data_gdf.dtypes}")
    # end_time = t.time()
    # Calculate the total time taken
    # computation_time = end_time - start_time
    # logger.info(
    # f"Conversion to vector layer processing time: {computation_time:.2f} seconds"
    # )
    return vector_data_gdf.to_json()

##################################################################################################################################
##################################################################################################################################
# The following functions are used in the Satellite_position class in the entity.py file

def get_elevation_angle(t, sat, loc):
    """
    Returns the elevation angle (degrees) of satellite with respect to the topocentric horizon.

    Args:
        t (:obj:`Time`): Time object of skyfield.timelib module
        sat (:obj:`EarthSatellite`): Skyview EarthSatellite object from skyfield.sgp4lib module
        loc (:obj:`GeographicPosition`): Geographic location on surface specified by latitude-longitude from skyfield.toposlib module

    Returns:
        float : alt.degrees
            Elevation angle (degrees) of satellite with respect to the topocentric horizon
    """
    difference = sat - loc
    topocentric = difference.at(t)
    # NOTE: Topos uses term altitude for what we are referring to as elevation
    alt, az, distance = topocentric.altaz()
    return alt.degrees

def compute_sensor_radius(altitude, min_elevation):
    """
    Computes the sensor radius for a satellite at current altitude given minimum elevation constraints.

    Args:
        altitude (float): Altitude (meters) above surface of the observation
        min_elevation (float): Minimum angle (degrees) with horizon for visibility

    Returns:
        float : sensor_radius
            The radius (meters) of the nadir pointing sensors circular view of observation
    """
    earth_equatorial_radius = 6378137.0
    earth_polar_radius = 6356752.314245179
    earth_mean_radius = (2 * earth_equatorial_radius + earth_polar_radius) / 3
    # rho is the angular radius of the earth viewed by the satellite
    sin_rho = earth_mean_radius / (earth_mean_radius + altitude)
    # eta is the nadir angle between the sub-satellite direction and the target location on the surface
    eta = np.degrees(np.arcsin(np.cos(np.radians(min_elevation)) * sin_rho))
    # calculate swath width half angle from trigonometry
    sw_HalfAngle = 90 - eta - min_elevation
    if sw_HalfAngle < 0.0:
        return 0.0
    return earth_mean_radius * np.radians(sw_HalfAngle)
