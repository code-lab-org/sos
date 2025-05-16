# This Script stores all the functions that are used in the simulator code
# The functions are stored in a separate file to make the main code more readable

import logging
import os
import sys

# Importing Libraries
from datetime import datetime, timedelta, timezone
from typing import List

import geopandas as gpd
import pandas as pd
from boto3.s3.transfer import TransferConfig
from shapely import Geometry
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
import time as t

from shapely import wkt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.sos_tools.aws_utils import AWSUtils
from src.sos_tools.data_utils import DataUtils

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


def compute_opportunity(
    const: List[Satellite],
    time: datetime,
    duration: timedelta,
    requests: List[dict],
) -> gpd.GeoSeries:
    # filter requests
    start_time = t.time()
    # logger.info(f"{type(const)},{const}")
    filtered_requests = requests
    # logger.info(f"Entered compute_opportunity,length of request is {len(filtered_requests)}, type of filtered request is,{type(filtered_requests)}")
    # logger.info(f"time :{type(time)}, duration: {type(duration)},combined: {type(time + duration)},tz info of time{time.tzinfo},tz info of combined{(time + duration).tzinfo}")
    time = time.replace(tzinfo=timezone.utc)
    end = (time + duration).replace(tzinfo=timezone.utc)   # end = (time + duration)
    filtered_requests = [
        request
        for request in requests
        if request.get("simulator_simulation_status") is None
        or pd.isna(request.get("simulator_simulation_status"))
    ]

    if filtered_requests:
        column_names = list(filtered_requests[0].keys())
        # logger.info(f"columns in filtered request{column_names}")

        # collect observation
        observation_results = pd.concat(
            [
                collect_multi_observations(request["point"], const, time, end)
                for request in filtered_requests
            ],
            ignore_index=True,
        ).sort_values(by="epoch", ascending=True)

        end_time = t.time()

        # Calculate the total time taken
        computation_time = end_time - start_time
        logger.info(f"Compute opportunity time: {computation_time:.2f} seconds")

        # observation_results = Parallel(n_jobs=-1)(
        #         delayed(collect_multi_observations)(
        #             point, constellation, time, time + duration
        #         )
        #         for point in filtered_requests
        #     )
        if observation_results is not None and not observation_results.empty:
            # logger.info(f"Observation opportunity exist{time + duration}")
            id_to_eta = {
            request["point"].id: request["planner_final_eta"]
            for request in filtered_requests
            }
            # observations = pd.concat(observation_results, ignore_index=True).sort_values(by="epoch", ascending=True)
            return observation_results.iloc[0]
        return None
    else:
        return None
    

# Filter and format the observation results

def filter_and_sort_observations(df, sim_time,incomplete_ids):
    # Filter for observations with incomplete simulation status
    df = df[df["point_id"].isin(incomplete_ids)]
    # Ensure sim_time is timezone-aware and same tz as df
    if df['epoch'].dt.tz is not None and sim_time.tzinfo is None:
        sim_time = sim_time.replace(tzinfo=df['epoch'].dt.tz)

    # Step 1: Filter for observations within 1 minute of simulation time
    one_minute_later = sim_time + timedelta(minutes=1)
    mask = (df["epoch"] >= sim_time) & (df["epoch"] <= one_minute_later)
    filtered = df[mask]

    # Step 2: Sort by planner_final_eta descending
    sorted_filtered = filtered.sort_values(by="planner_final_eta", ascending=False)
    return sorted_filtered.iloc[0] if not sorted_filtered.empty else None


# Computing Groundtrack and formatting into a dataframe

def compute_ground_track_and_format(
    sat_object: Satellite, observation_time: datetime
) -> Geometry:
    logger.info(
        f"Computing ground track for {sat_object.name} at {observation_time}, type of observation time is {type(observation_time)}"
    )
    # results = collect_ground_track(sat_object, [observation_time], crs="spice")
    observation_time = observation_time.replace(tzinfo=timezone.utc)
    results = collect_ground_track(sat_object, [observation_time], crs="spice")
    logger.info(f"Length of results{len(results)},type of results{type(results)}")
    # Formatting the dataframe
    # results["geometry"].iloc[0]
    return results["geometry"].iloc[0]





# CALLBACK FUNCTIONS
# The data from the appender is a geodataframe, reading and converting that to a list of dictionaries


def read_master_file():
    print("Reading Master file")
    logger.info("Reading master file")
    start_time = t.time()
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
                "planner_final_eta": r["planner_final_eta"],
                "simulator_polygon_groundtrack": r["simulator_polygon_groundtrack"],
                "planner_geometry": r["geometry"]                
            },
            axis=1,
        ).tolist()
    else:
        print(f"Master file not found. Returning an empty list.")
        request_points = []

    end_time = t.time()
    # Calculate the total time taken
    computation_time = end_time - start_time
    logger.info(f"Read masterfile time: {computation_time:.2f} seconds")

    # logger.info(f"Type of requests file{type(request_points)}")
    # request_points= request_points.to_dict('records')
    return request_points


# this function is triggered by scenario time interval callback, it writes the daily output to a geojson file


def write_back_to_appender(source, time):
    # Establish connection to S3
    s3 = AWSUtils().client
    output_directory = os.path.join("outputs", source.app.app_name)
    data_utils = DataUtils()
    data_utils.create_directories([output_directory])

    logger.info(
        f"Checking if appender function is reading the source object{source},{len(source.requests)},{type(source.requests)},{type(time)},{time},{time.date()}"
    )
    logger.info(f"Simulator time on scenario callback{source.app.simulator._time}")
    # start_time = t.time()
    appender_data = process_master_file(source.requests)
    selected_json_data = pd.DataFrame(appender_data)
    # logger.info(f"Colums in selected json data{selected_json_data.columns}")
    logger.info(
        f"Type of simulator_polygon_groundtrack{type(selected_json_data['simulator_polygon_groundtrack'])}"
    )
    # selected_json_data['simulator_polygon_groundtrack'] = selected_json_data['simulator_polygon_groundtrack'].apply(wkt.loads)
    selected_json_data["simulator_polygon_groundtrack"] = selected_json_data[
        "simulator_polygon_groundtrack"
    ].apply(lambda x: wkt.loads(x) if isinstance(x, str) else x)
    gdf = gpd.GeoDataFrame(selected_json_data, geometry="simulator_polygon_groundtrack")
    gdf.to_file(f"outputs/master_simulator.geojson", driver="GeoJSON")
    logger.info(f"{source.app.app_name} sending message.")
    date_sim_time = source.app.simulator._time
    date_sim = str(date_sim_time.date()).replace("-", "")
    # gdf["simulator_completion_date"] = gdf["simulator_completion_date"].astype(str)
    # gdf["point"] = gdf["point"].astype(str)
    # gdf["planner_time"] = gdf["planner_time"].astype(str)

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
    # end_time = t.time()
    # Calculate the total time taken
    # computation_time = end_time - start_time
    logger.info(f"Write back to appender time: {computation_time:.2f} seconds")


# the master file from the appender is in geojson format, and the requests are in a list of dictionaries, writing back the completed requests to the master file(triggered on available requests callback)
def process_master_file(existing_request):
    logger.info(f"Processing master file")
    start_time = t.time()
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
                    # if  key == 'point':
                    #     unprocessed_request["simulator_id"] = value.id
                    #     unprocessed_request["planner_latitude"] = value.latitude
                    #     unprocessed_request["planner_longitude"] = value.longitude
                    # else:
                    unprocessed_request[key] = value
                #   unprocessed_request.update(request)  # Update only fields, don't replace dict

    end_time = t.time()
    # Calculate the total time taken
    computation_time = end_time - start_time
    logger.info(f"Process master file time: {computation_time:.2f} seconds")

    # Return the combined list of processed and unprocessed requests
    return master_processed + master_unprocessed


# Converting data to be comaptible to the message body vector layer format


def convert_to_vector_layer_format(visual_requests):
    start_time = t.time()
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
    logger.info(f"type of vector data gdf{vector_data_gdf.dtypes}")

    end_time = t.time()
    # Calculate the total time taken
    computation_time = end_time - start_time
    logger.info(f"Covnert to vector layer time: {computation_time:.2f} seconds")
    return vector_data_gdf.to_json()
