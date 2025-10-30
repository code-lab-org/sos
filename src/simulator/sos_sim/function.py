# This Script stores all the functions that are used in the simulator code
# The functions are stored in a separate file to make the main code more readable
import json
import logging
import os
import sys
import threading
import time as _time
from datetime import datetime, timedelta, timezone
from typing import List
import geopandas as gpd
import numpy as np
import pandas as pd
from boto3.s3.transfer import TransferConfig
from shapely import Geometry, wkt
from tatc.analysis import collect_ground_track, collect_multi_observations
from tatc.schemas import (
    Point,
    PointedInstrument,
    Satellite,
    SunSynchronousOrbit,
    WalkerConstellation,
)
from joblib import Parallel, delayed
from tatc.utils import swath_width_to_field_of_regard, swath_width_to_field_of_view

logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from src.simulator.constellation_config_files.schemas import VectorLayer
from src.sos_tools.aws_utils import AWSUtils
from src.sos_tools.data_utils import DataUtils


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
    # Computation time of this function
    # start_time = _time.perf_counter()
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
        # column_names = list(filtered_requests[0].keys())

        def collect_observations_for_request(request):
            try:
                return collect_multi_observations(request["point"], const, time, end)
            except Exception as e:
                print(f"Error processing request {request}: {e}")
                return pd.DataFrame()

        observation_results_list = Parallel(n_jobs=-1 if parallel_compute else 1)(
            delayed(collect_observations_for_request)(request)
            for request in filtered_requests
        )

        # # Remove any empty results
        observation_results_list = [df for df in observation_results_list if not df.empty]

        if observation_results_list:
            observation_results = pd.concat(
                observation_results_list, ignore_index=True
            ).sort_values(by="epoch", ascending=True)
        else:
            observation_results = pd.DataFrame()

        # observation_results = pd.concat(
        #     [
        #         collect_multi_observations(request["point"], const, time, end)
        #         for request in filtered_requests
        #     ],
        #     ignore_index=True,
        # ).sort_values(by="epoch", ascending=True)

        if observation_results is not None and not observation_results.empty:
            # logger.info(f"Observation opportunity exist{time + duration}")
            id_to_eta = {
                request["point"].id: request["planner_final_eta"]
                for request in filtered_requests
            }
            observation_results["planner_final_eta"] = observation_results[
                "point_id"
            ].map(id_to_eta)
            # observations = pd.concat(observation_results, ignore_index=True).sort_values(by="epoch", ascending=True)
            # end_time = _time.perf_counter()
            # computation_time = end_time - start_time
            # logger.info(f"Observation opportunity computation time: {computation_time:.2f} seconds")
            return observation_results
        return None
    else:
        return None


# Filter and format the observation results


def filter_and_sort_observations(df, sim_time, incomplete_ids, time_step_constr):

    # logger.debug(
    #     f"Filtering and sorting observations, type of df is {type(df)}, length of df is {len(df)}"
    # )
    # Filter for observations with incomplete simulation status
    df = df[df["point_id"].isin(incomplete_ids)]
    # Ensure sim_time is timezone-aware and same tz as df
    if df["epoch"].dt.tz is not None and sim_time.tzinfo is None:
        sim_time = sim_time.replace(tzinfo=df["epoch"].dt.tz)

    # Step 1: Filter for observations within 1 minute of simulation time
    time_step_later = sim_time + time_step_constr
    # logger.debug(
    #     f"Filtering observations, sim_time: {sim_time}, time_step_later: {time_step_later}, type of sim_time is {type(sim_time)}, type of time_step_later is {type(time_step_later)}"
    # )
    mask = (df["epoch"] >= sim_time) & (df["epoch"] <= time_step_later)
    filtered = df[mask]

    # logger.debug(
    #     f"Filtered observations, type of filtered is {type(filtered)}, length of filtered is {len(filtered)}"
    # )
    # Step 2: Sort by planner_final_eta descending
    sorted_filtered = filtered.sort_values(by="planner_final_eta", ascending=False)
    logger.debug(
        f"Sorted observations, type of sorted is {sorted_filtered}, length of sorted is {len(sorted_filtered)}"
    )
    return sorted_filtered.iloc[0] if not sorted_filtered.empty else None


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


def read_master_file(request_data=None) -> List[dict]:
    """
    Read the master file and convert it to a list of dictionaries.
    Returns:
        List[dict]: A list of dictionaries representing the master file data.
    """
    logger.info("Reading master file")
    # Computation time of this function
    start_time = _time.perf_counter()
    # logger.info(f"Type of request_data: {type(request_data)}")
    # logger.info(f"Request data :{request_data}")
    # output_filename = "outputs/master.geojson"
    if request_data is not None:
        # request_data = gpd.read_file(output_filename)
        request_points = request_data.apply(
            lambda r: {
                "point": Point(
                    id=r["simulator_id"],
                    latitude=r["planner_latitude"],
                    longitude=r["planner_longitude"],
                ),
                "simulator_id": r["simulator_id"],
                "planner_time": r["planner_time"],
                "planner_final_eta": r["planner_final_eta"],
                "planner_latitude": r["planner_latitude"],
                "planner_longitude": r["planner_longitude"],
                "simulator_expiration_date": r["simulator_expiration_date"],
                "simulator_expiration_status": r["simulator_expiration_status"],
                "simulator_simulation_status": r["simulator_simulation_status"],
                "simulator_completion_date": r["simulator_completion_date"],
                "simulator_satellite": r["simulator_satellite"],
                "simulator_polygon_groundtrack": r["simulator_polygon_groundtrack"],
                "planner_geometry": r["geometry"],
            },
            axis=1,
        ).tolist()
    else:
        print(f"Master data is empty. Returning an empty list.")
        request_points = []

    end_time = _time.perf_counter()

    # end_time = t.time()
    # Calculate the total time taken
    computation_time = end_time - start_time
    logger.info(f"Reading master file time: {computation_time:.2f} seconds")
    return request_points


# this function is triggered by scenario time interval callback, it writes the daily output to a geojson file
# def _write_back_to_appender_impl(thread_data):
#     """
#     Internal implementation of write_back_to_appender that does the actual work.
#     This runs in a background thread to avoid blocking the simulation.

#     Args:
#         thread_data (dict): Dictionary containing captured data from the main thread
#     """
#     logger.info("write_back_to_appender thread started.")
#     # Monotonic timer (avoid shadowing by aliasing time as _time)
#     _start = _time.perf_counter()

#     # Extract data from thread_data
#     requests = thread_data["requests"]
#     master_data = thread_data["master_data"]
#     app_name = thread_data["app_name"]
#     sim_time = thread_data["sim_time"]
#     callback_time = thread_data["callback_time"]
#     source = thread_data.get("source")
#     enable_uploads = thread_data.get(
#         "enable_uploads", True
#     )  # Default to True for backward compatibility

#     logger.info(
#         f"Background thread starting with {len(requests)} requests, app_name: {app_name}, sim_time: {sim_time}"
#     )

#     # Establish connection to S3
#     s3 = AWSUtils().client
#     output_directory = os.path.join("outputs", app_name)
#     data_utils = DataUtils()
#     data_utils.create_directories([output_directory])

#     logger.info(
#         f"Checking if appender function is reading the requests: {len(requests)}, type: {type(requests)}, callback_time: {callback_time}, sim_time: {sim_time}"
#     )

#     try:
#         # Build DataFrame of updated master requests
#         appender_data = process_master_file(requests, master_data)
#         logger.info(f"Processed master file, got {len(appender_data)} entries")

#         selected_json_data = pd.DataFrame(appender_data)
#         logger.info(f"Created DataFrame with shape: {selected_json_data.shape}")

#         # Vectorized WKT parsing for performance
#         if "simulator_polygon_groundtrack" in selected_json_data.columns:
#             col = selected_json_data["simulator_polygon_groundtrack"]
#             if not col.empty:
#                 mask = col.apply(lambda v: isinstance(v, str))
#                 if mask.any():
#                     selected_json_data.loc[mask, "simulator_polygon_groundtrack"] = (
#                         gpd.GeoSeries.from_wkt(col.loc[mask].astype(str))
#                     )

#         gdf = gpd.GeoDataFrame(
#             selected_json_data, geometry="simulator_polygon_groundtrack"
#         )
#         logger.info(f"Created GeoDataFrame with shape: {gdf.shape}")

#         # Write master file without index to reduce IO
#         gdf.to_file("outputs/master.geojson", driver="GeoJSON", index=False)
#         logger.info(f"Wrote master.geojson with {len(gdf)} records")

#         logger.info(f"{app_name} sending message.")
#         date_sim_time = sim_time
#         date_sim = date_sim_time.strftime("%Y%m%d")

#         # Ensure datetime dtype only if needed
#         if (
#             "simulator_completion_date" in gdf.columns
#             and not pd.api.types.is_datetime64_any_dtype(
#                 gdf["simulator_completion_date"]
#             )
#         ):
#             gdf["simulator_completion_date"] = pd.to_datetime(
#                 gdf["simulator_completion_date"], errors="coerce"
#             )

#         # Filter for the current simulation date
#         daily_gdf_filtered = gdf[
#             gdf["simulator_completion_date"].dt.date == sim_time.date()
#         ]

#         logger.info(
#             f"Filtered daily data: {len(daily_gdf_filtered)} records for date {sim_time.date()}"
#         )

#         current_simulation_date = os.path.join(
#             output_directory, str(date_sim_time.date())
#         )
#         data_utils.create_directories([current_simulation_date])
#         output_file = os.path.join(
#             current_simulation_date, f"simulator_output_{date_sim}.geojson"
#         )
#         daily_gdf_filtered.to_file(output_file, index=False)
#         logger.info(f"Wrote daily file: {output_file}")

#         # Upload to S3 only if uploads are enabled
#         if enable_uploads:
#             logger.info(f"Uploading file to S3: {output_file}")
#             # Use threaded multipart upload for speed
#             upload_cfg = TransferConfig(
#                 multipart_threshold=8 * 1024 * 1024,
#                 multipart_chunksize=8 * 1024 * 1024,
#                 max_concurrency=32,
#                 use_threads=True,
#             )
#             s3.upload_file(
#                 Bucket="snow-observing-systems",
#                 Key=output_file,
#                 Filename=output_file,
#                 Config=upload_cfg,
#             )
#             logger.info(f"Uploaded to S3: {output_file}")
#         else:
#             logger.info(f"Upload skipped (uploads disabled): {output_file}")

#         daily_gdf_filtered = daily_gdf_filtered[
#         [
#             "simulator_id",
#             "simulator_simulation_status",
#             "simulator_completion_date",
#             "simulator_satellite",
#             "simulator_polygon_groundtrack"
#         ]
#         ]
#         daily_gdf_filtered["simulator_completion_date"] = daily_gdf_filtered["simulator_completion_date"].astype(str)

#         selected_json_data = daily_gdf_filtered.to_json()
#         source.app.send_message(
#             app_name,
#             "simulator_daily",  # ["master", "selected"],
#             VectorLayer(vector_layer=selected_json_data).model_dump_json()
#         )

#         logger.info("Simulator sent message to appender")

#         elapsed = _time.perf_counter() - _start
#         logger.info(f"write_back_to_appender completed in {elapsed:.2f} seconds")

#     except Exception as e:
#         elapsed = _time.perf_counter() - _start
#         logger.error(
#             f"write_back_to_appender failed after {elapsed:.2f} seconds: {e}",
#             exc_info=True,
#         )


# def write_back_to_appender(source, time):
#     """
#     Write the processed data back to the appender and upload it to S3.
#     This function runs the actual work in a background thread to avoid blocking the simulation.

#     Args:
#         source: The source object containing the data to be processed.
#         time (datetime): The time at which to write back the data.
#     """
#     # Capture the current state of source.requests to avoid race conditions
#     # Make a deep copy of the requests list to ensure thread safety
#     import copy

#     captured_requests = copy.deepcopy(source.requests)
#     captured_app_name = source.app.app_name
#     captured_sim_time = source.app.simulator._time
#     master_data = copy.deepcopy(source.master_data)

#     logger.info(
#         f"Capturing data for background thread: {len(captured_requests)} requests, sim_time: {captured_sim_time}"
#     )

#     # Create a simple data container to pass to the thread
#     thread_data = {
#         "requests": captured_requests,
#         "app_name": captured_app_name,
#         "sim_time": captured_sim_time,
#         "master_data": master_data,
#         "callback_time": time,
#         "source": source,
#         "enable_uploads": getattr(
#             source, "enable_uploads", True
#         ),  # Default to True for backward compatibility
#     }

#     logger.info(f"Thread data prepared, length of requests: {len(thread_data['requests'])}")

#     # Create a daemon thread so it doesn't prevent program shutdown
#     thread = threading.Thread(
#         target=_write_back_to_appender_impl,
#         args=(thread_data,),
#         daemon=True,
#         name=f"write_back_to_appender-{time.strftime('%Y%m%d-%H%M%S')}",
#     )

#     logger.info(f"Starting write_back_to_appender in background thread for time {time}")
#     thread.start()

    # Return immediately - don't wait for the thread to complete
    # This allows the simulation to continue without blocking


# def process_master_file(existing_request,master_data) -> List[dict]:
#     """
#     Process the master file and update the existing requests with the new data.
#     Args:
#         existing_request (List[dict]): The existing requests to be updated.
#     Returns:
#         List[dict]: The updated list of requests.
#     """
#     logger.info(f"Processing master file.")
#     # Computing computation time of this function
#     start_time = _time.perf_counter()

#     master = read_master_file(master_data)
#     # Computation time of master_processed
#     master_processed_start = _time.perf_counter()
#     master_processed = [
#         request
#         for request in master
#         if request["simulator_simulation_status"] == "Completed"
#     ]
#     master_processed_end = _time.perf_counter()
#     logger.info(f"Processed master simulation status file in {master_processed_end - master_processed_start:.2f} seconds.")

#     master_unprocessed = [
#         request for request in master if request["simulator_simulation_status"] is None
#     ]
#     # Update unprocessed master entries with latest fields from existing_request by point match
#     for unprocessed_request in master_unprocessed:
#         for request in existing_request:
#             if request["point"] == unprocessed_request["point"]:
#                 for key, value in request.items():
#                     unprocessed_request[key] = value

#     end_time = _time.perf_counter()
#     computation_time = end_time - start_time
#     logger.info(f"Processed master file in {computation_time:.2f} seconds.")

#     # Return the combined list of processed and unprocessed requests
#     return master_processed + master_unprocessed


def convert_to_vector_layer_format(visual_requests):
    """
    Convert the visual requests to a GeoDataFrame and then to a GeoJSON format.
    Args:
        visual_requests (List[dict]): The list of visual requests to be converted.
    Returns:
        str: The GeoJSON representation of the visual requests.
    """
    vector_data = pd.DataFrame(visual_requests)
    # logger.info(f"data types and columns: {vector_data.dtypes}, {vector_data.columns}")
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
    return vector_data_gdf.to_json()


def message_to_geojson(body):
        """
        Converts a message body to a GeoDataFrame.

        Inputs:
            body (bytes): The message body to convert.

        Returns:
            GeoDataFrame: The GeoDataFrame created from the message
        """
        logger.info("Converting message body to GeoDataFrame.")
        body = body.decode("utf-8")
        logger.info("Decoding body completed")
        data = VectorLayer.model_validate_json(body)
        logger.info("Validating body completed")
        k = gpd.GeoDataFrame.from_features(
            json.loads(data.vector_layer)["features"], crs="EPSG:4326"
        )
        return k


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

##################################################################################################################################
##################################################################################################################################

# this function is triggered by scenario time interval callback, it writes the daily output to a geojson file
def _write_back_to_appender_impl(thread_data):
    """
    Internal implementation of write_back_to_appender that does the actual work.
    This runs in a background thread to avoid blocking the simulation.

    Args:
        thread_data (dict): Dictionary containing captured data from the main thread
    """
    logger.info("write_back_to_appender thread started.")
    # Monotonic timer (avoid shadowing by aliasing time as _time)
    _start = _time.perf_counter()

    # Extract data from thread_data
    requests = thread_data["requests"]
    # master_data = thread_data["master_data"]
    app_name = thread_data["app_name"]
    sim_time = thread_data["sim_time"]
    callback_time = thread_data["callback_time"]
    source = thread_data.get("source")
    enable_uploads = thread_data.get(
        "enable_uploads", True
    )  # Default to True for backward compatibility

    logger.info(
        f"Background thread starting with {len(requests)} requests, app_name: {app_name}, sim_time: {sim_time}"
    )

    # Establish connection to S3
    s3 = AWSUtils().client
    output_directory = os.path.join("outputs", app_name)
    data_utils = DataUtils()
    data_utils.create_directories([output_directory])

    logger.info(
        f"Checking if appender function is reading the requests: {len(requests)}, type: {type(requests)}, callback_time: {callback_time}, sim_time: {sim_time}"
    )

    try:
        # # Build DataFrame of updated master requests
        # appender_data = process_master_file(requests, master_data)
        # logger.info(f"Processed master file, got {len(appender_data)} entries")

        # computation time of this function
        start_time_gdf = _time.perf_counter()

        selected_data = pd.DataFrame(requests)
        selected_data = selected_data.drop(columns=["point"])
        logger.info(f"Selected json data column dtypes: {selected_data.dtypes}")
        # .to_json(orient="records")
        # logger.info(f"Created DataFrame with shape: {selected_data.shape}")

        # # Vectorized WKT parsing for performance
        # if "simulator_polygon_groundtrack" in selected_data.columns:
        #     col = selected_data["simulator_polygon_groundtrack"]
        #     if not col.empty:
        #         mask = col.apply(lambda v: isinstance(v, str))
        #         if mask.any():
        #             selected_data.loc[mask, "simulator_polygon_groundtrack"] = (
        #                 gpd.GeoSeries.from_wkt(col.loc[mask].astype(str))
        #             )

        gdf = gpd.GeoDataFrame(
            selected_data,
            geometry="simulator_polygon_groundtrack",
            crs="EPSG:4326"
        )
        # logger.info(f"Created GeoDataFrame with shape: {gdf.shape}")

        # # Write master file without index to reduce IO
        # gdf.to_file("outputs/master.geojson", driver="GeoJSON", index=False)
        # logger.info(f"Wrote master.geojson with {len(gdf)} records")

        # logger.info(f"{app_name} sending message.")
        date_sim_time = sim_time
        date_sim = sim_time.strftime("%Y%m%d")

        # Ensure datetime dtype only if needed
        if (
            "simulator_completion_date" in gdf.columns
            and not pd.api.types.is_datetime64_any_dtype(
                gdf["simulator_completion_date"]
            )
        ):
            gdf["simulator_completion_date"] = pd.to_datetime(
                gdf["simulator_completion_date"], errors="coerce"
            )

        # Filter for the current simulation date
        daily_gdf_filtered = gdf[
            gdf["simulator_completion_date"].dt.date == sim_time.date()
        ]

        logger.info(
            f"Filtered daily data: {len(daily_gdf_filtered)} records for date {sim_time.date()}"
        )

        current_simulation_date = os.path.join(
            output_directory, str(date_sim_time.date())
        )
        data_utils.create_directories([current_simulation_date])
        output_file = os.path.join(
            current_simulation_date, f"simulator_output_{date_sim}.geojson"
        )
        daily_gdf_filtered.to_file(output_file, index=False)
        logger.info(f"Wrote daily file: {output_file}")
        end_time_gdf = _time.perf_counter()
        logger.info(f"GeoDataFrame processing time: {end_time_gdf - start_time_gdf:.2f} seconds.")

        # Computation time for upload
        # start_time_upload = _time.perf_counter()

        # Upload to S3 only if uploads are enabled
        if enable_uploads:
            logger.info(f"Uploading file to S3: {output_file}")
            # Use threaded multipart upload for speed
            upload_cfg = TransferConfig(
                multipart_threshold=8 * 1024 * 1024,
                multipart_chunksize=8 * 1024 * 1024,
                max_concurrency=32,
                use_threads=True,
            )
            s3.upload_file(
                Bucket="snow-observing-systems",
                Key=output_file,
                Filename=output_file,
                Config=upload_cfg,
            )
            logger.info(f"Uploaded to S3: {output_file}")
        else:
            logger.info(f"Upload skipped (uploads disabled): {output_file}")

        # Computation time for appender message
        start_time_appender = _time.perf_counter()

        daily_gdf_filtered = daily_gdf_filtered[
        [
            "simulator_id",
            "simulator_simulation_status",
            "simulator_completion_date",
            "simulator_satellite",
            "simulator_polygon_groundtrack"
        ]
        ]
        daily_gdf_filtered["simulator_completion_date"] = daily_gdf_filtered["simulator_completion_date"].astype(str)

        selected_json_data = daily_gdf_filtered.to_json()
        source.app.send_message(
            app_name,
            "simulator_daily",  # ["master", "selected"],
            VectorLayer(vector_layer=selected_json_data).model_dump_json()
        )

        logger.info("Simulator sent message to appender")

        elapsed = _time.perf_counter() - _start
        elapsed_appender = _time.perf_counter() - start_time_appender
        logger.info(f"Total appender message time: {elapsed_appender:.2f} seconds.")        
        logger.info(f"write_back_to_appender completed in {elapsed:.2f} seconds")

    except Exception as e:
        elapsed = _time.perf_counter() - _start
        logger.error(
            f"write_back_to_appender failed after {elapsed:.2f} seconds: {e}",
            exc_info=True,
        )


def write_back_to_appender(source, time):
    """
    Write the processed data back to the appender and upload it to S3.
    This function runs the actual work in a background thread to avoid blocking the simulation.

    Args:
        source: The source object containing the data to be processed.
        time (datetime): The time at which to write back the data.
    """
    # Capture the current state of source.requests to avoid race conditions
    # Make a deep copy of the requests list to ensure thread safety
    import copy

    captured_requests = copy.deepcopy(source.requests)
    captured_app_name = source.app.app_name
    captured_sim_time = source.app.simulator._time
    # master_data = copy.deepcopy(source.master_data)

    logger.info(
        f"Capturing data for background thread: {len(captured_requests)} requests, sim_time: {captured_sim_time}"
    )

    # Create a simple data container to pass to the thread
    thread_data = {
        "requests": captured_requests,
        "app_name": captured_app_name,
        "sim_time": captured_sim_time,
        # "master_data": master_data,
        "callback_time": time,
        "source": source,
        "enable_uploads": getattr(
            source, "enable_uploads", True
        ),  # Default to True for backward compatibility
    }

    logger.info(f"Thread data prepared, length of requests: {len(thread_data['requests'])}")

    # Create a daemon thread so it doesn't prevent program shutdown
    thread = threading.Thread(
        target=_write_back_to_appender_impl,
        args=(thread_data,),
        daemon=True,
        name=f"write_back_to_appender-{time.strftime('%Y%m%d-%H%M%S')}",
    )

    logger.info(f"Starting write_back_to_appender in background thread for time {time}")
    thread.start()

    # Return immediately - don't wait for the thread to complete
    # This allows the simulation to continue without blocking
