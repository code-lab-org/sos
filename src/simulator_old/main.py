import datetime
import json
import logging
from datetime import datetime, timedelta, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
from constellation_config_files.schemas import VectorLayer
from nost_tools.application_utils import ShutDownObserver
from nost_tools.config import ConnectionConfig
from nost_tools.managed_application import ManagedApplication
from nost_tools.observer import Observer
from nost_tools.simulator import Mode, Simulator
from tatc.analysis import collect_ground_track, collect_multi_observations
from tatc.schemas import (
    Point,
    PointedInstrument,
    SunSynchronousOrbit,
    WalkerConstellation,
)
from tatc.utils import swath_width_to_field_of_regard, swath_width_to_field_of_view

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger()


class Environment(Observer):
    """
    *The Environment object class inherits properties from the Observer object class in the NOS-T tools library*

    Attributes:
        app (:obj:`ManagedApplication`): An application containing a test-run namespace, a name and description for the app, client credentials, and simulation timing instructions
        grounds (:obj:`DataFrame`): DataFrame of ground station information including groundId (*int*), latitude-longitude location (:obj:`GeographicPosition`), min_elevation (*float*) angle constraints, and operational status (*bool*)
    """

    def __init__(self, app):  # , grounds):
        self.app = app
        self.first_run = None
        self.flag = None

    def const(self):  # initialize_snowglobe_constellation(self):
        # logger.info("Initializing SnowGlobe constellation.")
        roll_angle = (30 + 33.5) / 2
        roll_range = 33.5 - 30
        start = datetime(2019, 1, 1, tzinfo=timezone.utc)
        self.constellation = WalkerConstellation(
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
                    along_track_field_of_view=swath_width_to_field_of_view(
                        555e3, 50e3, 0
                    ),
                    cross_track_field_of_view=roll_range
                    + swath_width_to_field_of_view(555e3, 50e3, roll_angle),
                    is_rectangular=True,
                )
            ],
        )
        satellites = self.constellation.generate_members()
        self.satellite_dict = {sat.name: sat for sat in satellites}
        # logger.info("Initializing SnowGlobe constellation successfully completed.")

    # def user_request(self):  # filter_master_file(self):

    #     if self.first_run:
    #         logger.info("Using master file sent by planner.")
    #         self.req = self.gdf
    #         start = self.app._sim_start_time
    #         stop = self.app._sim_stop_time
    #         current = self.app.simulator._time
    #         previous = current - timedelta(1)
    #         previous_reformat = str(previous.date()).replace("-", "")

    #         if previous.date() > start.date():

    #             logger.info(f"Length of Request: {len(self.req)}.")

    #             try:
    #                 logger.info("Reading previous local master GeoJSON file.")
    #                 self.previous = gpd.read_file(
    #                     f"local_master_{previous_reformat}.geojson"
    #                 )
    #             except:
    #                 logger.info("Reading previous previous local master GeoJSON file.")
    #                 previous_previous = previous - timedelta(1)
    #                 previous_previous_reformat = str(previous_previous.date()).replace(
    #                     "-", ""
    #                 )
    #                 self.previous = gpd.read_file(
    #                     f"local_master_{previous_previous_reformat}.geojson"
    #                 )
    #             self.req = (
    #                 self.req.set_index("geometry")
    #                 .combine_first(self.previous.set_index("geometry"))
    #                 .reset_index()
    #             )
    #             self.req = gpd.GeoDataFrame(self.req, crs=4326, geometry="geometry")
    #             logger.info(f"Length of Request After Merge: {len(self.req)}.")

    #     else:
    #         logger.info("Reading local master GeoJSON file.")
    #         self.req = gpd.read_file(f"local_master_{self.date}.geojson")
    #     self.filtered_req = self.req[
    #         self.req["simulator_simulation_status"].isna()
    #         | (self.req["simulator_simulation_status"] == "None")
    #     ]

    def user_request(self):

        if self.first_run:
            logger.info("Using master file sent by planner.")
            self.req = self.gdf
            start = self.app._sim_start_time
            stop = self.app._sim_stop_time
            current = self.app.simulator._time
            previous = current - timedelta(1)
            previous_reformat = str(previous.date()).replace("-", "")

            if previous.date() > start.date():

                logger.info(f"Length of Request: {len(self.req)}.")

                try:
                    logger.info("Reading previous local master GeoJSON file.")
                    self.previous = gpd.read_file(
                        f"local_master_{previous_reformat}.geojson"
                    )
                except:
                    logger.info("Reading previous previous local master GeoJSON file.")
                    previous_previous = previous - timedelta(1)
                    previous_previous_reformat = str(previous_previous.date()).replace(
                        "-", ""
                    )
                    self.previous = gpd.read_file(
                        f"local_master_{previous_previous_reformat}.geojson"
                    )
                self.req = (
                    self.req.set_index("geometry")
                    .combine_first(self.previous.set_index("geometry"))
                    .reset_index()
                )
                self.req = gpd.GeoDataFrame(self.req, crs=4326, geometry="geometry")
                logger.info(f"Length of Request After Merge: {len(self.req)}.")

        else:
            logger.info("Reading local master GeoJSON file.")
            self.req = gpd.read_file(f"local_master_{self.date}.geojson")
        self.filtered_req = self.req[
            self.req["simulator_simulation_status"].isna()
            | (self.req["simulator_simulation_status"] == "None")
        ]

    def opportunity(self, start_time=None):
        logger.info("Calculating opportunity.")
        # self.const()
        # self.user_request()
        start_time = start_time or self._time
        end = start_time + timedelta(1)  # 2)
        start_time = start_time.replace(tzinfo=timezone.utc)
        end = end.replace(tzinfo=timezone.utc)
        combined_results = pd.DataFrame()
        for index, row in self.filtered_req.iterrows():
            loc = Point(
                id=row["simulator_id"],
                latitude=row["planner_latitude"],
                longitude=row["planner_longitude"],
            )
            results = collect_multi_observations(
                loc, self.constellation, start_time, end
            )
            combined_results = pd.concat([combined_results, results], ignore_index=True)
        self.combined_results = combined_results.sort_values(by="epoch", ascending=True)
        logger.info("Calculating opportunity successfully completed.")

    def message_to_geojson(self, body):
        # Convert the message body to a string then to a GeoDataFrame
        body = body.decode("utf-8")
        data = VectorLayer.model_validate_json(body)

        # Convert the string back to a GeoJSON object
        return gpd.GeoDataFrame.from_features(
            json.loads(data.vector_layer)["features"], crs="EPSG:4326"
        )

    def on_appender(self, ch, method, properties, body):
        # Initialization
        self.first_run = True

        # Create a GeoDataFrame from the GeoJSON object
        self.gdf = self.message_to_geojson(body)

        # Initialization
        self.const()  # previously initialize_snowglobe_constellation()
        self.user_request()  # previously filter_master_file()
        self.first_run = False

        # Define values for the first run
        self._time = self._next_time = self._init_time = self.app.simulator._time
        self.date = str(self._time.date()).replace("-", "")
        self._next_time = self._time + timedelta(days=1)

        # while self._time < self._init_time + self._duration:
        while self._time < self._next_time:
            flag = 0
            # logger.info(f"Current time: {self._time}")
            # logger.info(f"Next time: {self._next_time}")

            # # updating user requests - read filtered rows
            # self.user_request()

            # self.filtered_req = gpd.GeoDataFrame()
            # Error handler
            if self.filtered_req.empty:
                logging.info("No observations available. Skipping to next time step.")
                # Can use this condition to reset the master file
                self._time = self._next_time
                continue

            # Calculate opportunity
            self.opportunity()

            # self.combined_results = gpd.GeoDataFrame()
            # Error handler
            if self.combined_results.empty:
                logging.error(
                    "Combined results is empty! No observations until next time step! Skipping to next"
                )
                self._time = self._next_time
                continue

            counter = 0

            self.observation_time = self.combined_results["epoch"].iloc[
                counter
            ]  # latest possible observation
            self.id = self.combined_results["point_id"].iloc[
                counter
            ]  # point id for the above observation
            self.coord = self.combined_results["geometry"].iloc[
                counter
            ]  # location for the observation
            self.sat = self.combined_results["satellite"].iloc[
                counter
            ]  # satellite collecting the observation
            prev_observation_time = None
            len_rs = len(self.combined_results)

            while self.observation_time < self._next_time:
                len_rs = len(self.combined_results)
                # logger.info(counter)

                # Ensuring no more than 1 observation is collected at a time
                if prev_observation_time is None:
                    prev_observation_time = self.observation_time
                elif self.observation_time <= (
                    prev_observation_time + timedelta(minutes=1)
                ) and (counter <= len_rs):
                    counter += 1
                    self.observation_time = self.combined_results["epoch"].iloc[counter]
                    self.id = self.combined_results["point_id"].iloc[counter]
                    self.sat = self.combined_results["satellite"].iloc[counter]
                    continue
                elif counter >= len_rs:
                    break

                # Constellation Capacity Logic
                if (np.random.rand() <= 0.25) & (counter <= len_rs):
                    counter += 1
                    self.observation_time = self.combined_results["epoch"].iloc[counter]
                    self.id = self.combined_results["point_id"].iloc[counter]
                    self.sat = self.combined_results["satellite"].iloc[counter]
                elif counter >= len_rs:
                    break
                else:

                    prev_observation_time = self.observation_time
                    self.req  # reads the requests file

                    # Formatting
                    self.req["simulator_completion_date"] = pd.to_datetime(
                        self.req["simulator_completion_date"], errors="coerce"
                    )  # Ensure simulator_simulation_status is datetime
                    self.req["simulator_simulation_status"] = self.req[
                        "simulator_simulation_status"
                    ].astype(
                        str
                    )  # Ensure simulation_status is string
                    self.req["simulator_satellite"] = self.req[
                        "simulator_satellite"
                    ].astype(str)
                    # format time as required in gejson file
                    # self.observation_time
                    # t = self.observation_time.replace(tzinfo=None)

                    # Assigning values to master file
                    self.req.loc[
                        self.req.simulator_id == self.id, "simulator_completion_date"
                    ] = self.observation_time
                    self.req.loc[
                        self.req.simulator_id == self.id, "simulator_simulation_status"
                    ] = "Completed"
                    # req.loc[req.id == self.id, 'request_status'] = 'Completed'
                    self.req.loc[
                        self.req.simulator_id == self.id, "simulator_satellite"
                    ] = self.sat

                    # Groundtrack information
                    sat_object = self.satellite_dict.get(self.sat)
                    results = collect_ground_track(
                        sat_object, [self.observation_time]  # , crs="spice"
                    )
                    self.req.loc[
                        self.req.simulator_id == self.id,
                        "simulator_polygon_groundtrack",
                    ] = results["geometry"].iloc[0]

                    # Values to be sent to appender ###  EMMANUEL PLACEHOLDER
                    # Define the data structure for the GeoDataFrame

                    # data_appender = {
                    #     "attribute": [
                    #         "simulator_id",
                    #         "simulator_completion_date",
                    #         "simulator_simulation_status",
                    #         "simulator_satellite",
                    #         "geometry",
                    #     ],
                    #     "value": [
                    #         self.id,
                    #         self.observation_time,
                    #         "Completed",
                    #         self.sat,
                    #         results["geometry"].iloc[0],
                    #     ],
                    # }

                    if not self.req.empty:
                        # Set values greater than current time to None
                        self.req.loc[
                            pd.to_datetime(
                                self.req["simulator_completion_date"]
                            ).dt.date
                            > self.app.simulator._time.date(),
                            "simulator_simulation_status",
                        ] = np.nan  # None
                        self.req.loc[
                            pd.to_datetime(
                                self.req["simulator_completion_date"]
                            ).dt.date
                            > self.app.simulator._time.date(),
                            "simulator_expiration_date",
                        ] = pd.NaT
                        self.req.loc[
                            pd.to_datetime(
                                self.req["simulator_completion_date"]
                            ).dt.date
                            > self.app.simulator._time.date(),
                            "simulator_satellite",
                        ] = np.nan  # None
                        self.req.loc[
                            pd.to_datetime(
                                self.req["simulator_completion_date"]
                            ).dt.date
                            > self.app.simulator._time.date(),
                            "simulator_polygon_groundtrack",
                        ] = np.nan  # None
                        self.req.loc[
                            pd.to_datetime(
                                self.req["simulator_completion_date"]
                            ).dt.date
                            > self.app.simulator._time.date(),
                            "simulator_completion_date",
                        ] = pd.NaT

                        self.req.to_file(
                            f"local_master_{self.date}.geojson", driver="GeoJSON"
                        )
                        logger.info("Successfully updated local master GeoJSON file.")
                    self.first_run = False

                    # Regenerating observations with respect to updated list
                    # calling user_request and opportunity will now exclude the entries processed above(sompleted status) and generate new list
                    self.user_request()
                    counter = 0
                    flag += 1

                    # Error handler
                    if self.filtered_req.empty:
                        logging.info(
                            "No observations available. Skipping to next time step."
                        )
                        # Can use this condition to reset the master file
                        self._time = self._next_time
                        continue

                    self.opportunity(self.observation_time + timedelta(minutes=1))

                    if self.combined_results.empty:
                        logging.error(
                            "Combined results is empty! No observations until next time step! Skipping to next"
                        )
                        self._time = self._next_time
                        continue

                    # self.rs = self.combined_results
                    self.observation_time = self.combined_results["epoch"].iloc[counter]
                    self.id = self.combined_results["point_id"].iloc[counter]
                    self.sat = self.combined_results["satellite"].iloc[counter]
                    len_rs = len(self.combined_results)

            if flag > 0:
                self.user_request()
                # Filter data for each day (self.time)
                date = str(self._time.date()).replace("-", "")
                file_name = f"Simulator_Output_{date}.geojson"
                # logger.info(f">>>TIME: {self._time}")

                filtered_data = self.req[
                    pd.to_datetime(self.req["simulator_completion_date"]).dt.date
                    == pd.to_datetime(self._time).date()
                ]
                filtered_data.to_file(file_name, driver="GeoJSON")
                flag = 0

            self._time = self._next_time

            if not self.req.empty:
                # Convert the clipped GeoDataFrame to GeoJSON and send as message
                selected_data = self.req[
                    [
                        "planner_final_eta",
                        "planner_time",
                        "simulator_simulation_status",
                        "simulator_satellite",
                        "geometry",
                    ]
                ]  # Add simulation time
                selected_data["planner_time"] = selected_data["planner_time"].astype(
                    str
                )
                selected_json_data = selected_data.to_json()
                self.app.send_message(
                    "planner",
                    "selected",
                    VectorLayer(vector_layer=selected_json_data).model_dump_json(),
                )
                logger.info("(SELECTED) Publishing message successfully completed.")

    def on_appender_alternative(self, ch, method, properties, body):

        # Define values for the first run
        self._time = self._next_time = self._init_time = self.app.simulator._time
        self.date = str(self._time.date()).replace("-", "")
        self._next_time = self._time + timedelta(days=1)
        flag = 0

        date = self.app.simulator._time
        date = str(date.date()).replace("-", "")

        # All Tracks
        geojson_path = f"local_master_{date}.geojson"
        gdf = gpd.read_file(geojson_path)

        # Convert the clipped GeoDataFrame to GeoJSON and send as message
        selected_data = gdf[
            [
                # "planner_final_eta",
                "planner_time",
                "simulator_simulation_status",
                "simulator_completion_date",
                "simulator_satellite",
                "geometry",
            ]
        ]  # Add simulation time
        selected_data["planner_time"] = selected_data["planner_time"].astype(str)
        # Assuming selected_data is your DataFrame
        selected_data["simulator_completion_date"] = pd.to_datetime(
            selected_data["simulator_completion_date"]
        )
        selected_data["simulator_completion_date"] = selected_data[
            "simulator_completion_date"
        ].dt.strftime("%Y-%m-%d %H:%M:%S")
        selected_json_data = selected_data.to_json()
        self.app.send_message(
            "planner",
            "selected",
            VectorLayer(vector_layer=selected_json_data).model_dump_json(),
        )
        logger.info("(SELECTED) Publishing message successfully completed.")

    def on_change(self, source, property_name, old_value, new_value):
        if property_name == Simulator.PROPERTY_MODE and new_value == Mode.EXECUTING:
            print(old_value, new_value)


def main():
    # Load config
    config = ConnectionConfig(yaml_file="sos.yaml")

    # Define the simulation parameters
    NAME = "simulator"

    # create the managed application
    app = ManagedApplication(NAME)

    # initialize the Environment object class
    environment = Environment(app)

    # add the environment observer to monitor simulation for switch to EXECUTING mode
    app.simulator.add_observer(Environment(app))

    # add a shutdown observer to shut down after a single test case
    app.simulator.add_observer(ShutDownObserver(app))

    # start up the application on PREFIX, publish time status every 10 seconds of wallclock time
    app.start_up(
        config.rc.simulation_configuration.execution_parameters.general.prefix, config
    )

    app.add_message_callback("appender", "master", environment.on_appender)


if __name__ == "__main__":
    main()
