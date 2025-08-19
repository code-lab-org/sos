import logging
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd
from constellation_config_files.schemas import SatelliteStatus, VectorLayer

# from geojson_pydantic import Polygon, MultiPolygon
# from joblib import Parallel, delayed
from nost_tools import Application, Entity
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
        # logger.info(
        #     f"entering tick time {self._time}, {len(self.requests)}, next time {self._next_time}"
        # )
        # Set all the tick operations here
        self.observation_collected = None
        # logger.info(
        #     f"Type of date {type(self._time)} and self.last_observation_time {type(self.last_observation_time)}"
        # )

        t1 = self._time.replace(tzinfo=None)
        t2 = self.last_observation_time.replace(tzinfo=None) + timedelta(seconds=30)
        # logger.info(f"t1 {t1} and t2 {t2}")

        if t1 > t2:

            if self.possible_observations is not None:
                # logger.info(
                # #     f"Number of possible observations {len(self.possible_observations)}"
                # )
                self.observation_collected = filter_and_sort_observations(
                    self.possible_observations,
                    self._time,
                    self.incomplete_requests,
                    timedelta(seconds=30),
                )

            if self.observation_collected is not None:

                if (
                    np.random.rand() <= 1.0
                ):  # Simulate a 75% chance of collecting an observation
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
                    # logger.info(f"Observation {self.observation_collected}")
                    # logger.info(f"Observation type {type(self.observation_collected)}")

                    self.next_requests = self.requests.copy()

                    # update next_requests to reflect collected observation
                    for row in self.next_requests:
                        if row["point"].id == self.observation_collected["point_id"]:
                            # row["point"] = TATC_Point(id = self.observation_collected["point_id"], latitude = self.observation_collected['geometry'].y, longitude = self.observation_collected['geometry'].x)
                            # Point(id=r["simulator_id"], latitude=r["planner_latitude"], longitude=r["planner_longitude"])
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

                            # logger.info(f"Type of collected time {type(self.observation_collected['epoch'])}")

                            self.last_observation_time = self.observation_collected[
                                "epoch"
                            ]
                            # logger.info(
                            #     f"Type of collected time {type(self.last_observation_time)}"
                            # )

                            # Remove from incomplete_requests
                            if row["point"].id in self.incomplete_requests:
                                self.incomplete_requests.remove(row["point"].id)

                        # logger.info(f"Type of polygon groundtrack{type(row['simulator_polygon_groundtrack'])}")

                    # Visualization
                    # write a function to convert the self.next request to json format to send to the cesium application
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

    def tock(self):
        # logger.info("entering tock time")
        super().tock()
        # logger.info("entering tock time")
        if self.observation_collected is not None:
            self.requests = self.next_requests

        if self.new_request_flag:
            logger.info("Requests received.")
            self.requests = process_master_file(self.requests)
            self.incomplete_requests = [
                r["point"].id
                for r in self.requests
                if r.get("simulator_simulation_status") is None
                or pd.isna(r.get("simulator_simulation_status"))
            ]
            self.possible_observations = compute_opportunity(
                list(self.constellation.values()),
                self._time,
                timedelta(days=1),
                self.requests,
            )
            # logger.info(
            #     f"Number of possible observations {len(self.possible_observations)}"
            # )
            self.new_request_flag = False

    def message_received_from_appender(self, ch, method, properties, body):
        # logger.info(f"Message succesfully received at {self.app.simulator._time}")
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
