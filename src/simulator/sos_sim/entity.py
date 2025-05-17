import logging
from datetime import datetime, timedelta
from typing import List

import numpy as np
from constellation_config_files.schemas import VectorLayer
from nost_tools import Application, Entity
from tatc.schemas import Satellite as TATC_Satellite

from .function import (  # update_requests,; filter_requests,
    compute_ground_track_and_format,
    compute_opportunity,
    convert_to_vector_layer_format,
    process_master_file,
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
        self.new_request_flag = None

    def on_appender(self):
        self.new_request_flag = True

    def tick(self, time_step: timedelta):
        super().tick(time_step)
        # Set all the tick operations here
        self.observation_collected = compute_opportunity(
            list(self.constellation.values()), self._time, time_step, self.requests
        )

        if self.observation_collected is not None:
            if np.random.rand() <= 0.75:

                # get the satellite that collected the observation
                satellite = self.constellation[self.observation_collected["satellite"]]
                # Call the groundtrack function
                self.observation_collected["ground_track"] = (
                    compute_ground_track_and_format(
                        satellite, self.observation_collected["epoch"]
                    )
                )
                self.next_requests = self.requests.copy()
                # update next_requests to reflect collected observation
                for row in self.next_requests:
                    if row["point"].id == self.observation_collected["point_id"]:
                        # row["point"] = TATC_Point(id = self.observation_collected["point_id"], latitude = self.observation_collected['geometry'].y, longitude = self.observation_collected['geometry'].x)
                        # Point(id=r["simulator_id"], latitude=r["planner_latitude"], longitude=r["planner_longitude"])
                        row["simulator_simulation_status"] = "Completed"
                        row["simulator_completion_date"] = self.observation_collected[
                            "epoch"
                        ]
                        row["simulator_satellite"] = self.observation_collected[
                            "satellite"
                        ]
                        row["simulator_polygon_groundtrack"] = (
                            self.observation_collected["ground_track"]
                        )

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
            else:
                self.observation_collected = None

    def tock(self):
        super().tock()
        if self.observation_collected is not None:
            self.requests = self.next_requests

        if isinstance(self.app.simulator._time, str):
            current_date = self.app.simulator._time.replace("-", "")  # Already a string
        else:
            current_date = self.app.simulator._time.date().strftime("%Y%m%d")

        if self.new_request_flag:
            logger.info("requests received")
            self.requests = process_master_file(self.requests)
            self.new_request_flag = False

    def message_received_from_appender(self, ch, method, properties, body):
        logger.info(f"Message succesfully received at {self.app.simulator._time}")
        self.on_appender()
