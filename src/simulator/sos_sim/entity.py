import logging
from datetime import datetime, timedelta
from typing import List
import numpy as np
from constellation_config_files.schemas import VectorLayer
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.framelib import itrs

# from geojson_pydantic import Polygon, MultiPolygon
# from joblib import Parallel, delayed
from nost_tools import Entity, Application
import numpy as np
import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import Point
from skyfield.api import wgs84
from tatc.analysis import collect_ground_track, collect_observations
from tatc.schemas import Satellite as TATC_Satellite, Point as TATC_Point
from .function import (
    compute_opportunity,
    # update_requests,
    Snowglobe_constellation,
    compute_ground_track_and_format,
    compute_opportunity,
    convert_to_vector_layer_format,
    process_master_file,
    convert_to_vector_layer_format,
    filter_and_sort_observations
)

logger = logging.getLogger(__name__)
np.random.seed(0)


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
    ):
        super().__init__()
        # save initial values
        self.init_constellation = constellation
        self.init_requests = requests
        self.app = application
        # declare state variables
        self.constellation = None
        self.requests = []
        self.incomplete_requests = []
        self.next_requests = None
        self.possible_observations = None
        self.last_observation_time = None
        self.observation_collected = None
        self.new_request_flag = False

    def initialize(self, init_time: datetime):
        super().initialize(init_time)
        # initialize state variables
        self.constellation = {sat.name: sat for sat in self.init_constellation}
        self.requests = self.init_requests.copy()
        self.next_requests = None
        self.last_observation_time = datetime.min
        self.observation_collected = None
        self.new_request_flag = None

    def on_appender(self):
        self.new_request_flag = True

    def tick(self, time_step: timedelta):
        
        super().tick(time_step)       
        logger.info(f"entering tick time {self._time}, {len(self.requests)}, next time {self._next_time}")        
        # Set all the tick operations here
        self.observation_collected = None
        logger.info(f"Type of date {type(self._time)} and self.last_observation_time {type(self.last_observation_time)}")         
        
        t1 = self._time.replace(tzinfo=None)
        t2 = self.last_observation_time.replace(tzinfo=None) + timedelta(seconds=30)
        logger.info(f"t1 {t1} and t2 {t2}")

        if t1 > t2:

            if self.possible_observations is not None:
                logger.info(f"Number of possible observations {len(self.possible_observations)}")
                self.observation_collected = filter_and_sort_observations(
                    self.possible_observations, self._time, self.incomplete_requests,timedelta(seconds=30))     
            
            if self.observation_collected is not None:

                if np.random.rand() <= 1:
                    # Get the satellite that collected the observation
                    satellite = self.constellation[self.observation_collected["satellite"]]
                    # Call the groundtrack function
                    self.observation_collected["ground_track"] = compute_ground_track_and_format(
                        satellite, self.observation_collected["epoch"]
                    )
                    logger.info(f"Observation {self.observation_collected}")
                    # logger.info(f"Observation type {type(self.observation_collected)}")
                    
                    self.next_requests = self.requests.copy()
                    
                    # update next_requests to reflect collected observation
                    for row in self.next_requests:
                        if row["point"].id == self.observation_collected["point_id"]:
                            # row["point"] = TATC_Point(id = self.observation_collected["point_id"], latitude = self.observation_collected['geometry'].y, longitude = self.observation_collected['geometry'].x)
                            #Point(id=r["simulator_id"], latitude=r["planner_latitude"], longitude=r["planner_longitude"])
                            row["simulator_simulation_status"] = "Completed"
                            row["simulator_completion_date"] = self.observation_collected["epoch"]
                            row["simulator_satellite"] = self.observation_collected["satellite"]
                            row["simulator_polygon_groundtrack"] = self.observation_collected["ground_track"]

                            # logger.info(f"Type of collected time {type(self.observation_collected['epoch'])}")

                            self.last_observation_time = self.observation_collected["epoch"]
                            logger.info(f"Type of collected time {type(self.last_observation_time)}")

                            # Remove from incomplete_requests
                            if row["point"].id in self.incomplete_requests:
                                self.incomplete_requests.remove(row["point"].id)

                        # logger.info(f"Type of polygon groundtrack{type(row['simulator_polygon_groundtrack'])}")

                    # Visualization
                    # write a function to convert the self.next request to json format to send to the cesium application
                    vector_data_json = convert_to_vector_layer_format(self.next_requests)                
                    # Sending message to visualization
                    self.app.send_message(
                        self.app.app_name,
                            "selected",
                                        VectorLayer(vector_layer=vector_data_json).model_dump_json(),
                                    )
                    logger.info("(SELECTED) Publishing message successfully completed.")
                # else:
                #     self.observation_collected = None

    def tock(self):
        logger.info("entering tock time")
        super().tock()
        # logger.info("entering tock time")
        if self.observation_collected is not None:          
            self.requests = self.next_requests

        if self.new_request_flag:
            logger.info("requests received")
            self.requests = process_master_file(self.requests)
            self.incomplete_requests = [
            r["point"].id
            for r in self.requests
            if r.get("simulator_simulation_status") is None or pd.isna(r.get("simulator_simulation_status"))
            ]
            self.possible_observations = compute_opportunity(
            list(self.constellation.values()), self._time, timedelta(days=1), self.requests)
            logger.info(f"Number of possible observations {len(self.possible_observations)}")      
            self.new_request_flag = False

    def message_received_from_appender(self, ch, method, properties, body):
        logger.info(f"Message succesfully received at {self.app.simulator._time}")
        self.on_appender()


class SatelliteVisualization(Entity):
    """
    Visualizes the satellites on Cesium.
    """
    ts = load.timescale()
    PROPERTY_POSITION = "position"

    def __init__(self, 
        constellation: List[TATC_Satellite], 
        application: Application):
        super().__init__()
        self.names = {sat.name: sat for sat in constellation}
        
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

        