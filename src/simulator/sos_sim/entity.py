from datetime import datetime, timedelta
from typing import List
import logging
import numpy as np
import os
import pandas as pd
import sys
import threading
import time as _time
from constellation_config_files.schemas import SatelliteStatus, VectorLayer
from nost_tools import Application, Entity
from nost_tools.simulator import Mode, Simulator
from nost_tools.observer import Observer
from pyproj import Transformer
from skyfield.api import load
from skyfield.framelib import itrs
from tatc.analysis import collect_ground_track, collect_orbit_track
from tatc.schemas import Satellite as TATC_Satellite


from .function import (  # update_requests,
    compute_ground_track_and_format,
    compute_opportunity,
    compute_sensor_radius,
    convert_to_vector_layer_format,    
    filter_and_sort_observations,
    message_to_geojson,
    Daily_random_value,
    process_master_file
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.sos_tools.aws_utils import AWSUtils

logger = logging.getLogger(__name__)

class Collect_Observations(Entity):
    """
    Reports the next observation opportunity and
    records observations when collected.
    """

    # defining class constants
    PROPERTY_OBSERVATION = "observation_collected"

    def __init__(
        self,
        constellation: List[TATC_Satellite],
        requests: List[dict],
        application: Application,
        const_capacity: float = 1.0,
        time_interval: float = 1.0,
        s3_variable = None,
        enable_uploads=None
    ):
        super().__init__()
        # save initial values
        self.init_constellation = constellation
        self.init_requests = requests
        self.app = application
        self.constellation_capacity = const_capacity
        self.time_between_observations = int(time_interval)

        if s3_variable is not None:
            self.s3_bucket = s3_variable
        else:
            self.s3_bucket = AWSUtils().client
            

        # Flag to control S3 uploads - check environment variable if not explicitly set        
        if enable_uploads is None:
            self.enable_uploads = os.environ.get("ENABLE_UPLOADS", "true").lower() in (
                "true",
                "1",
                "yes",
            )
        else:
            self.enable_uploads = enable_uploads
        # declare state variables
        self.constellation = None
        self.requests = []
        self.incomplete_requests = []
        self.next_requests = None
        self.possible_observations = None
        self.last_observation_time = None
        self.observation_collected = None
        self.observation_collected_flag = False
        self.new_request_flag = False
        self.master_data = None
        self.seed_value = 0 # Seed value for daily random value generation
        self.rng_cache = {}  # Cache for daily random values
        self.daily_random_value = None  # Daily random value object
        # self.sim_stop_flag = False
        # Threading state variables
        self.master_file_processing = False
        self.processed_requests = None
        self.incomplete_requests_processed = None
        self.possible_observations_processed = None
        self.master_file_lock = threading.Lock()

    def initialize(self, init_time: datetime):
        super().initialize(init_time)
        # initialize state variables
        self.constellation = {sat.name: sat for sat in self.init_constellation}
        self.requests = self.init_requests.copy() if self.init_requests else []
        self.next_requests = None
        self.last_observation_time = datetime.min
        self.observation_collected = None
        self.new_request_flag = None

    def on_change(self, source, property_name, old_value, new_value):
        """
        Forward simulator time changes to this entity's observers.
        """
        # Check if the simulator is notifying time updates
        if property_name == Simulator.PROPERTY_TIME:
            # Forward the same event to any observers attached to this entity
            self.notify_observers(property_name, old_value, new_value) 

    def on_appender(self):
        self.new_request_flag = True

    def tick(self, time_step: timedelta):  
        # logger.info("Entering tick time")
        super().tick(time_step)
        # Set all the tick operations here
        # logger.info("Simulation stop time is %s and type is %s", self.sim_stop_time, type(self.sim_stop_time))
        self.observation_collected = None
        # Converting simulation time and last observation time to naive datetime for comparison
        t1 = self.get_time().replace(tzinfo=None)
        t2 = self.last_observation_time.replace(tzinfo=None) + timedelta(seconds=self.time_between_observations)       

        if t1 > t2:

            if self.possible_observations is not None:
                # logger.info("Self.possible_observations length is %d", len(self.possible_observations))

                self.observation_collected = filter_and_sort_observations(
                    self.possible_observations,
                    self._time,
                    self.incomplete_requests,
                    timedelta(seconds=self.time_between_observations),
                )

                if self.observation_collected is not None:
                    # Generate or retrieve the daily random value
                    # Generate and cache the random value for the new day

                    self.daily_random_value = Daily_random_value(
                        seed_value=self.seed_value,
                        min_value=0.0,
                        max_value=1.0,
                        rng_cache=self.rng_cache
                    )
                    if (
                        self.daily_random_value <= self.constellation_capacity
                    ): 

                        self.observation_collected_flag = True
                        # logger.info("Daily random value is %f", self.daily_random_value)
                        # logger.info("Constellation capacity is %f", self.constellation_capacity)
                        # logger.info("Observation collected") 
                        # Simulate a x% chance of collecting an observation
                        # Get the satellite that collected the observation
                        satellite = self.constellation[
                            self.observation_collected["satellite"]
                        ]
                        # Call the groundtrack function
                        self.observation_collected["ground_track"] = (
                            compute_ground_track_and_format(
                                satellite, self.observation_collected["epoch"]
                            )
                        )
                        # logger.info("self.rquest is %d", len(self.requests) if self.requests is not None else 0)
                        self.next_requests = self.requests.copy() if self.requests is not None else []
                        # logger.info("Updated next_requests with current requests, length is %d", len(self.next_requests) if self.next_requests is not None else 0   )
                        # logger.info("self.rquest is %d", len(self.requests) if self.requests is not None else 0)
                        # Update next_requests to reflect collected observation
                        for row in self.next_requests:
                            if row["point"].id == self.observation_collected["point_id"]:                            
                                row["simulator_simulation_status"] = "Completed"
                                row["simulator_completion_date"] = (
                                    self.observation_collected["epoch"]
                                )
                                row["simulator_satellite"] = self.observation_collected[
                                    "satellite"
                                ]
                                row["simulator_polygon_groundtrack"] = (
                                    self.observation_collected["ground_track"]
                                )

                                self.last_observation_time = self.observation_collected[
                                    "epoch"
                                ]
                                # Remove from incomplete_requests
                                if row["point"].id in self.incomplete_requests:
                                    self.incomplete_requests.remove(row["point"].id)

                        # Visualization
                        # Write a function to convert the self.next request to json format to send to the cesium application
                        # Execute if self. next_requests is not None

                        if self.next_requests:
                            vector_data_json = convert_to_vector_layer_format(
                                self.next_requests
                            )
                            # Sending message to visualization
                            self.app.send_message(
                                self.app.app_name,
                                "selected",
                                VectorLayer(vector_layer=vector_data_json).model_dump_json(),
                            )
                    # logger.info("(SELECTED) Publishing message successfully completed.")
                # else:
                #     self.observation_collected = None

    def _process_master_file_impl(self, thread_data):
        """
        Internal implementation of master file processing that runs in a background thread.

        Args:
            thread_data (dict): Dictionary containing captured data from the main thread
        """
        import time as _time

        _start = _time.perf_counter() 
        current_time = thread_data["current_time"]
        constellation_values = thread_data["constellation_values"]
        new_requests = thread_data["new_requests"]
        existing_completed_requests = thread_data["existing_completed_requests"]
        logger.info("New requests length in background thread is %d and type %s", len(new_requests), type(new_requests))


        try:
            # Import here to avoid circular imports in thread
            from .function import read_master_file

            logger.info("Starting master file processing in background thread,length of self.master_data is %d", len(self.master_data))

            processed_requests = process_master_file(new_requests,existing_completed_requests)

            # logger.info("Master file read and processed with %d requests", len(processed_requests))

            # Also compute incomplete requests and opportunities in background
            incomplete_requests = [
                r["point"].id
                for r in processed_requests
                if r.get("simulator_simulation_status") is None
                or pd.isna(r.get("simulator_simulation_status"))
            ]

            # Compute opportunities (this is also a heavy operation)
            # Computation time of this function
            start_time = _time.perf_counter()
            possible_observations = compute_opportunity(
                constellation_values,
                current_time,
                timedelta(days=1),
                processed_requests,
            )
            end_time = _time.perf_counter()
            computation_time = end_time - start_time
            logger.info("Opportunity computation time: %.2f seconds", computation_time)

            # logger.info("Computed %d possible observations", len(possible_observations))

            # Thread-safe update of the result
            with self.master_file_lock:
                self.processed_requests = processed_requests
                self.incomplete_requests_processed = incomplete_requests
                self.possible_observations_processed = possible_observations
                self.master_file_processing = False
                self.new_request_flag = False

            elapsed = _time.perf_counter() - _start
            logger.info(
                f"Master file processing (including opportunities) completed in background thread in {elapsed:.2f} seconds"
            )

        except Exception as e:
            elapsed = _time.perf_counter() - _start
            logger.error(
                f"Master file processing failed in background thread after {elapsed:.2f} seconds: {e}",
                exc_info=True,
            )

            # Ensure we reset the processing flag even on error
            with self.master_file_lock:
                self.master_file_processing = False

    def tock(self):
        # logger.info("entering tock time")
        super().tock()

        # logger.info("Length of requests at tock: %d", len(self.requests) if self.requests is not None else 0)
        # logger.info("Length of next_requests at tock: %d", len(self.next_requests) if self.next_requests is not None else 0)
        # logger.info("Master file processing status at tock: %s", self.master_file_processing)
        # logger.info("entering tock time")
        if self.observation_collected_flag:    
            # logger.info("Observation collected at tock: %s", self.observation_collected)    
            self.requests = self.next_requests
            self.observation_collected_flag = False  # Reset the flag after updating
            # logger.info("Updated requests from next_requests at tock after observation collected)")

        # logger.info("Length of requests after updating from next_requests at tock: %d", len(self.requests) if self.requests is not None else 0)
        
        # Start background processing if new requests arrived and not already processing
        if self.new_request_flag and not self.master_file_processing:
            logger.info(
                "Requests received. Starting background master file processing."
            )

            # Filtering the completed requests from self.request, this will be passed to the processing function to capture completed request by the time new requests arrive

            completed_requests = [
            r
            for r in self.requests
            if r.get("simulator_simulation_status") == "Completed"
            ]

            # Capture current state for thread safety
            import copy

            thread_data = {
                "new_requests": self.master_data,
                "existing_completed_requests": copy.deepcopy(completed_requests),
                "current_time": self._time,
                "constellation_values": list(self.constellation.values()),
            }

            # Mark as processing
            with self.master_file_lock:
                self.master_file_processing = True

            # Start background thread
            thread = threading.Thread(
                target=self._process_master_file_impl,
                args=(thread_data,),
                daemon=True,
                name=f"master_file_processing-{self._time.strftime('%Y%m%d-%H%M%S')}",
            )

            logger.info(
                f"Starting master file processing in background thread at {self._time}"
            )
            thread.start()

            # Don't reset new_request_flag here - let it be reset when processing completes        

        # Check if background processing completed
        if self.master_file_processing is False and self.processed_requests is not None:
            with self.master_file_lock:
                if self.processed_requests is not None:
                    logger.info("Applying processed master file results")
                    self.requests = self.processed_requests
                    logger.info("Number of requests after processing: %d", len(self.requests))
                    self.incomplete_requests = self.incomplete_requests_processed
                    self.possible_observations = self.possible_observations_processed

                    # Clear the processed results
                    self.processed_requests = None
                    self.incomplete_requests_processed = None
                    self.possible_observations_processed = None
                    self.new_request_flag = False

       
    def message_received_from_appender(self, ch, method, properties, body):
        logger.info(f"Message succesfully received at {self.app.simulator._time}")
        self.master_data = message_to_geojson(body)
        logger.info(f"Master data received with {len(self.master_data)} records and type {type(self.master_data)}")
        self.on_appender()


class SatelliteVisualization(Entity):
    """
    Visualizes the satellites on Cesium.
    """

    PROPERTY_POSITION = "position"

    def __init__(self, constellation: List[TATC_Satellite], application: Application):
        super().__init__()
        self.names = {sat.name: sat for sat in constellation}
        self.ts = load.timescale()
        self.app = application

    def initialize(self, init_time: datetime):
        """
        Initilizes the entity with the given start time.
        """
        super().initialize(init_time)

    def tick(self, time_step: timedelta):
        """
        Calculate and update the parameters required to visualize satellite positions.
        """
        super().tick(time_step)
        # logger.info(f"Tick time: {self._time}")

        for i, sat in enumerate(self.names.values()):

            self.id = i
            self.sat_name = sat.name
            # self.position = sat.get_position(self._time)
            self.point_obj = collect_orbit_track(sat, [self._time]).iloc[0].geometry
            self.point_obj_ecef = (
                collect_orbit_track(sat, [self._time], coordinates="ecef")
                .iloc[0]
                .geometry
            )

            logger.info(
                f" Satellite : {self.sat_name},  Time : {self._time}, point_obj_ecef: {self.point_obj_ecef}"
            )
            self.lon = self.point_obj.x
            self.lat = self.point_obj.y
            self.alt = self.point_obj.z
            self.velocity = (
                sat.orbit.to_tle()
                .as_skyfield()
                .at(self.ts.from_datetime(self._time))
                .frame_xyz_and_velocity(itrs)[1]
                .m_per_s
            )
            self.state = True
            self.swath = 50e3  # Example swath value in m
            self.ecef = [
                self.point_obj_ecef.x,
                self.point_obj_ecef.y,
                self.point_obj_ecef.z,
            ]
            geom = (
                collect_ground_track(sat, [self._time], crs="spice")
                .iloc[0]
                .geometry.centroid
            )
            lon_target, lat_target = geom.x, geom.y
            geo_to_ecef = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
            x, y, z = geo_to_ecef.transform(lon_target, lat_target, 0)
            self.ecef_target = [x, y, z]
            # self.ecef = [
            #     -532818.7563538807,
            #     5509596.270545596,
            #     4158666.755320652,
            # ]  # Example ECEF coordinates
            # self.ecef_target = [
            #     -490368.3475900224,
            #     5070546.733131086,
            #     3825230.5237126607,
            # ]
            self.radius = compute_sensor_radius(self.alt, 0)

            # logger.info(f"ECEF position for {i}: {self.ecef}")

            # logger.info(f"Publishing satellite position at {self._time}")

            self.app.send_message(
                self.app.app_name,
                "location",
                SatelliteStatus(
                    id=self.id,
                    name=self.sat_name,
                    # latitude=self.lat,
                    # longitude=self.lon,
                    # altitude=self.alt,
                    latitude=self.lat,
                    longitude=self.lon,
                    altitude=self.alt,
                    radius=self.radius,
                    velocity=self.velocity,
                    state=self.state,
                    swath=self.swath,
                    time=self._time,
                    ecef=self.ecef,
                    target=self.ecef_target,
                ).model_dump_json(),
            )

            # logger.info("Satellite position published successfully.")

            # if i == 2:
            #     # your test code here
            #     print(sat)
            # break

    def tock(self):
        """
        Publish the satellite position to the application.
        """
        super().tock()
        # logger.info(f"Publishing satellite position at {self._time}")
        # logger.info(f"tock after satellite position")

        # self.app.send_message(
        #         self.app.app_name,
        #         "location",
        #         SatelliteStatus(
        #             id=self.id,
        #             name=self.sat_name,
        #             latitude=self.lat,
        #             longitude=self.lon,
        #             altitude=self.alt,
        #             radius=self.radius,
        #             velocity=self.velocity,
        #             state=self.state,
        #             swath=self.swath,
        #             time=self._time,
        #             ecef=self.ecef,
        #         ).model_dump_json(),
        #     )

        # logger.info("Satellite position published successfully.")

class RandomValueGenerator(Observer):
    """
    This object class inherits from Observer and generates random values each day.
    """
    def __init__(self, app):
        self.app = app

    def detect_level_change(self, new_value, old_value, level):
        """
        Detect a change in the level of the time value (day, week, or month).

        Args:
            new_value (datetime): New time value
            old_value (datetime): Old time value
            level (str): Level of time value to detect changes ('day', 'week', or 'month')

        Returns:
            bool: True if the level has changed, False otherwise
        """
        if level == "day":
            return new_value.date() != old_value.date()
        elif level == "week":
            return new_value.isocalendar()[1] != old_value.isocalendar()[1]
        elif level == "month":
            return new_value.month != old_value.month
        else:
            raise ValueError("Invalid level. Choose from 'day', 'week', or 'month'.")
    
    def on_change(self, source, property_name, old_value, new_value):
        """
        Callback when simulation properties change.

        Args:
            source: The object that changed
            property_name (str): Name of the property that changed
            old_value: Previous value
            new_value: New value
        """
        # Only respond to time changes when simulation is executing
        if (
            property_name == Simulator.PROPERTY_TIME
            and source.app.simulator.get_mode() == Mode.EXECUTING
            and new_value is not None
            and self.detect_level_change(new_value, old_value, "day")
        ):
            # Update the seed value based on the new day
            source.seed_value = int(new_value.strftime("%Y%m%d"))
