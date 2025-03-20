from datetime import datetime, timedelta,timezone
import logging
from typing import List, Tuple
from constellation_config_files.schemas import VectorLayer

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
    # filter_requests,
    read_master_file,
    process_master_file
)

# from .schemas import Request, Observation

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
    ):
        super().__init__()
        # save initial values
        self.init_constellation = constellation
        self.init_requests = requests
        self.app = application

        # declare state variables
        self.constellation = None
        self.requests = []
        self.next_requests = None
        self.observation_collected = None
        self.new_request_flag = False

    def initialize(self, init_time: datetime):
        super().initialize(init_time)

        # initialize state variables
        self.constellation = {sat.name: sat for sat in self.init_constellation}
        self.requests = self.init_requests.copy()
        self.next_requests = None
        self.observation_collected = None
        self.new_request_flag  = None

    def on_appender(self):      
        self.new_request_flag = True

    def tick(self, time_step: timedelta):
        super().tick(time_step)
        # logger.info("entering tick time",self._time,len(self.requests),"next time",self._next_time)
        logger.info(f"entering tick time {self._time}, {len(self.requests)}, next time {self._next_time}")

        # Set all the tick operations here

        self.observation_collected = compute_opportunity(
           list(self.constellation.values()), self._time, time_step, self.requests
        )
        
        if self.observation_collected is not None:
            if np.random.rand() <= 0.75:
                # get the satellite that collected the observation
                satellite = self.constellation[self.observation_collected["satellite"]]
                # Call the groundtrack function
                self.observation_collected["ground_track"] = compute_ground_track_and_format(
                    satellite, self.observation_collected["epoch"]
                )
                self.next_requests = self.requests.copy()
                
                # update next_requests to reflect collected observation
                for row in self.next_requests:
                    if row["point"].id == self.observation_collected["point_id"]:
                        row["point"] = TATC_Point(id = self.observation_collected["point_id"], latitude = self.observation_collected['geometry'].y, longitude = self.observation_collected['geometry'].x)
                        #Point(id=r["simulator_id"], latitude=r["planner_latitude"], longitude=r["planner_longitude"])
                        row["simulator_simulation_status"] = "Completed"
                        row["simulator_completion_date"] = self.observation_collected["epoch"]
                        row["simulator_satellite"] = self.observation_collected["satellite"]
                        row["simulator_polygon_groundtrack"] = self.observation_collected["ground_track"]

                    logger.info(f"Type of polygon groundtrack{type(row['simulator_polygon_groundtrack'])}")

            else:
                self.observation_collected = None

    def tock(self):
        logger.info("entering tock time")
        super().tock()
        logger.info("entering tock time")
        if self.observation_collected is not None:
            logger.info("Notifying Observers")
            # self.notify_observers(
            #     self.PROPERTY_OBSERVATION,
            #     None,
            #     self.observation_collected,
            # )
            # update requests
            self.requests = self.next_requests  
        else: logger.info("No observation collected")          

        if isinstance(self.app.simulator._time, str):
            current_date = self.app.simulator._time.replace("-", "")  # Already a string
        else:
            current_date = self.app.simulator._time.date().strftime("%Y%m%d")        

        if self.new_request_flag:
            logger.info("requests received")
            # self.requests = read_master_file(current_date)
            # self.requests = read_master_file()
            self.requests = process_master_file(self.requests)
            self.new_request_flag = False

        # This code should execute only when message is received from the appender
        # self.new_request_flag should be set to 1 by the observer in the appender
        
        # # check for new requests
        # if self.new_requests is not None:
        #     for request in self.new_requests:
        #         self.requests.append(request)
        #     self.new_requests = None

    # def message_received_from_appender(self, client, userdata, message):
    def message_received_from_appender(self, ch, method, properties, body):
        # handle message received
        # self.app.add_message_callback("appender", "master", self.on_appender)
        logger.info("Message succesfully received")
        self.on_appender()  
    
       

    

    
        