import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
import geopandas as gpd
import numpy as np
import pandas as pd
from boto3.s3.transfer import TransferConfig
from constellation_config_files.schemas import VectorLayer
from nost_tools.application_utils import ShutDownObserver
from nost_tools.configuration import ConnectionConfig
from nost_tools.managed_application import ManagedApplication
from nost_tools.observer import Observer
from nost_tools.simulator import Mode, Simulator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.sos_tools.aws_utils import AWSUtils
from src.sos_tools.data_utils import DataUtils
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class Environment(Observer):
    """
    *The Environment object class inherits properties from the Observer object class in the NOS-T tools library*
    Attributes:
        app (:obj:`ManagedApplication`): An application containing a test-run namespace, a name and description for the app, client credentials, and simulation timing instructions
        grounds (:obj:`DataFrame`): DataFrame of ground station information including groundId (*int*), latitude-longitude location (:obj:`GeographicPosition`), min_elevation (*float*) angle constraints, and operational status (*bool*)
    """
    def __init__(self, app):
        self.app = app
        self.counter = 0
        self.master_components = gpd.GeoDataFrame()
        self.master_gdf = gpd.GeoDataFrame()
        self.visualize_selected = False  # True
        self.current_simulation_date = None
        self.output_directory = os.path.join("outputs", self.app.app_name)
        self.data_utils = DataUtils()
        self.data_utils.create_directories([self.output_directory])

    def add_prefix_to_columns(self, gdf, prefix):
        """
        Adds a prefix to each column name in the GeoDataFrame, except for the 'geometry' column.
        Inputs:
            gdf (GeoDataFrame): The GeoDataFrame whose columns will be prefixed.
            prefix (str): The prefix to add to each column name.

        Returns:
            GeoDataFrame: The GeoDataFrame with each column name prefixed
        """
        gdf.columns = [
            prefix + col if col != "geometry" else col for col in gdf.columns
        ]
        return gdf

    def add_columns(self, gdf):
        """
        Adds columns to the GeoDataFrame that will be filled by the simulator.
        Inputs:
            gdf (GeoDataFrame): The GeoDataFrame to which the columns will be added.
        Returns:
            GeoDataFrame: The GeoDataFrame with the additional columns added.
        """
        gdf["simulator_simulation_status"] = None  # None
        gdf["simulator_completion_date"] = pd.NaT
        # logger.info(f"Type planner time {type(gdf['planner_time'])}")
        gdf["simulator_expiration_date"] = pd.to_datetime(gdf["planner_time"]) + timedelta(days=2)
        # logger.info(f"Type planner time after conversion{type(gdf['planner_time'])}")
        gdf["simulator_expiration_status"] = None  # None
        gdf["simulator_satellite"] = None
        gdf["simulator_polygon_groundtrack"] = None
        gdf["planner_latitude"] = gdf["planner_centroid"].y
        gdf["planner_longitude"] = gdf["planner_centroid"].x
        gdf["planner_centroid"] = gdf["planner_centroid"].to_wkt()    
        logger.info(f"Computing simulator expiration status")
        gdf["simulator_expiration_date"] = gdf["simulator_expiration_date"].astype(str)

        return gdf

    def reorder_columns(self, gdf):
        """
        Reorders the columns of the GeoDataFrame to a specific order.

        Inputs:
            gdf (GeoDataFrame): The GeoDataFrame whose columns will be reordered.

        Returns:
            GeoDataFrame: The GeoDataFrame with the columns reordered.
        """
        gdf = gdf[
            [
                "simulator_id",
                "planner_time",
                "planner_final_eta",
                "planner_latitude",
                "planner_longitude",
                "simulator_expiration_date",
                "simulator_simulation_status",
                "simulator_completion_date",
                # "collected_within_last_3_days",
                "simulator_satellite",
                "simulator_polygon_groundtrack",
                "geometry",
            ]
        ]
        return gdf

    def add_last_observation_collected_time(self, gdf):
        """
        Adds a column for the last observation collected time to the GeoDataFrame.
        Inputs:
            gdf (GeoDataFrame): The GeoDataFrame to which the column will be added.

        Returns:
            GeoDataFrame: The GeoDataFrame with the additional column added.
        """
        # Create a geodataframe with geometry column and the last observation collected time with unique values of gdf
        gdf["simulator_completion_date"] = pd.to_datetime(gdf["simulator_completion_date"])      
        gdf_filtered = gdf[gdf["simulator_simulation_status"].notna()]  
        gdf_unique = (
            gdf_filtered.sort_values("simulator_completion_date")
            .drop_duplicates(subset=["geometry"], keep="last")
            .reset_index(drop=True)
        )
        gdf_unique.to_file(
            "last_collected_testing.geojson",
            driver="GeoJSON",
        )
        # Compute recent completion flag
        current_date = pd.Timestamp(self.app.simulator._time.date()).tz_localize("UTC")

        # logger.info("📅 simulator_completion_date dtype: %s", gdf_unique["simulator_completion_date"].dtype)
        # logger.info("🕒 current_date type: %s", type(current_date))
        # logger.info("🕒 current_date value: %s", current_date)

        if gdf_unique["simulator_completion_date"].dt.tz is None:
            gdf_unique["simulator_completion_date"] = gdf_unique["simulator_completion_date"].dt.tz_localize("UTC")

        gdf_unique["collected_within_last_3_days"] = (
                (current_date - gdf_unique["simulator_completion_date"]) <= pd.Timedelta(days=5)
        ).fillna(False)
        
        # Merge the collected flag back to original gdf by geometry
        
        gdf = gdf.merge(
            gdf_unique[["geometry", "collected_within_last_3_days"]],
            on="geometry",
            how="left"
        )
        # Fill any unmatched geometries with False
        gdf["collected_within_last_3_days"] = gdf["collected_within_last_3_days"].fillna(False)
        # logger.info(f"Gdf with last 3 days column populated,{gdf}")
        gdf.to_file(
            "outputs/master_last_collected_testing.geojson",
            driver="GeoJSON",
        )

        return gdf


    def process_component(self, component_gdf):
        """
        Inputs:
            component_gdf (GeoDataFrame): The GeoDataFrame of the component to process.
            counter (int): The counter to use for assigning unique IDs to the component.

        Returns:
            GeoDataFrame: The processed GeoDataFrame of the component.
        """
        logger.info("Processing component GeoJSON.")
        component_gdf["centroid"] = component_gdf.centroid
        component_gdf = self.add_prefix_to_columns(component_gdf, "planner_")
        component_gdf = self.add_columns(component_gdf)
        component_gdf["simulator_id"] = range(
            self.counter, self.counter + len(component_gdf)
        )        
        component_gdf = self.reorder_columns(component_gdf)
        component_gdf = component_gdf.to_crs(epsg=4326)
        logger.info("Processing component GeoJSON successfully completed.")
        return component_gdf

    def remove_duplicates(self):
        """
        Removes duplicate rows from the master GeoDataFrame.

        Returns:
            GeoDataFrame: The master GeoDataFrame with duplicates
        """
        completed_rows = self.master_gdf[
            self.master_gdf["simulator_simulation_status"] == "Completed"
        ]
        none_rows = self.master_gdf[
            self.master_gdf["simulator_simulation_status"].isna()
        ]
        most_recent_none_rows = none_rows.loc[
            none_rows.groupby("geometry")["planner_time"].idxmax()
        ]
        self.master_gdf = pd.concat(
            [completed_rows, most_recent_none_rows], ignore_index=True
        )

    def message_to_geojson(self, body):
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
        # k = gpd.GeoDataFrame.from_features(
        #     json.loads(data.vector_layer)["features"], crs="EPSG:4326"
        # )
        # logger.info(f"Message body successfully converted to GeoDataFrame. {type(k)}")

        features = json.loads(data.vector_layer).get("features", [])
        if not features:
            logger.warning("No features found in vector_layer — returning empty GeoDataFrame.")
            k = gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")
        else:
            k = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        # logger.info(f"Message body successfully converted to GeoDataFrame. {type(k)} with {len(k)} rows")
        logger.info("Message body successfully converted to GeoDataFrame.")

        return k

        # return gpd.GeoDataFrame.from_features(
        #     json.loads(data.vector_layer)["features"], crs="EPSG:4326" )

        

    def on_planner(self, ch, method, properties, body):
        """
        Responds to messages from planner application

        Inputs:
            ch (Channel): The channel on which the message was received.
            method (Method): The method used to receive the message.
            properties (Properties): The properties of the message.
            body (bytes): The body of the message.

        """
        logger.info("entering appender _on_planner")
        # Establish connection to S3
        s3 = AWSUtils().client

        # Convert the message body to a GeoDataFrame
        component_gdf = self.message_to_geojson(body)

        # Process the component GeoDataFrame
        component_gdf = self.process_component(component_gdf)
        # self.master_components.append(component_gdf)
        self.master_components = pd.concat([self.master_components, component_gdf], ignore_index=True)
        # logger.info(f"Master components type {self.master_components.dtypes}")
        # self.concatenated_master_components = pd.concat(
        #     self.master_components, ignore_index=True
        # )
        self.counter += len(component_gdf)
        # min_value = component_gdf["simulator_id"].min()
        # max_value = component_gdf["simulator_id"].max()
        
        # self.master_gdf = pd.concat(self.master_components, ignore_index=True)
        self.master_gdf = self.master_components.copy()
        self.master_gdf = self.add_last_observation_collected_time(self.master_gdf)
        self.master_gdf_with_duplicates = self.master_gdf.copy()
        self.remove_duplicates()
        date = self.app.simulator.get_time()  # Use a public getter instead of accessing _time directly
        date_new_format = str(date.date()).replace("-", "")
        self.current_simulation_date = os.path.join(
            self.output_directory, str(date.date())
        )
        self.data_utils.create_directories([self.current_simulation_date])
        output_file = os.path.join(
            self.current_simulation_date, f"appender_master_{date_new_format}.geojson"
        )
        self.master_gdf.to_file(
            output_file,
            driver="GeoJSON",
        )
        s3.upload_file(
            Bucket="snow-observing-systems",
            Key=output_file,
            Filename=output_file,
            Config=TransferConfig(use_threads=False),
        )
        self.master_gdf_with_duplicates.to_file("outputs/master.geojson", driver="GeoJSON")
        logger.info("Master geosjon file created")
        
        # Converting to str for json conversion
        datetime_cols = [
        "planner_time",
        "simulator_expiration_date",
        "simulator_completion_date"
        ]

        for col in datetime_cols:
            self.master_gdf[col] = self.master_gdf[col].astype(str)
            
        selected_json_data = self.master_gdf.to_json()
        self.app.send_message(
            self.app.app_name,
            "master",  # ["master", "selected"],
            VectorLayer(vector_layer=selected_json_data).model_dump_json(),
        )
        if self.visualize_selected:
            self.app.send_message(
                "planner",
                "selected",
                VectorLayer(vector_layer=selected_json_data).model_dump_json(),
            )
        logger.info(f"{self.app.app_name} sent message. at {self.app.simulator._time}")

    def on_simulator(self, ch, method, properties, body):
        """
        Responds to messages from simulator application(Updates updated columns in master gdf)

        Inputs:
            ch (Channel): The channel on which the message was received.
            method (Method): The method used to receive the message.
            properties (Properties): The properties of the message.
            body (bytes): The body of the message.

        """
        logger.info("Message Received from Simulator")
        component_gdf = self.message_to_geojson(body)
        # logger.info(f"dataframe from simulator type {component_gdf.dtypes}")

        if component_gdf.empty:
            logger.warning("Received empty message from simulator. Skipping update.")
            return

        # Select safe columns
        safe_cols = ["simulator_id", "simulator_simulation_status", "simulator_completion_date", "simulator_satellite"]
        safe_cols = [col for col in safe_cols if col in component_gdf.columns]
        component_gdf_clean = component_gdf[safe_cols].dropna(subset=["simulator_id"])
        component_gdf_clean['simulator_completion_date'] = pd.to_datetime(
            component_gdf_clean['simulator_completion_date'], errors='coerce'
        )

        # Updating values in self.master_components based on simulator_id
        self.master_components.set_index("simulator_id", inplace=True)
        component_gdf_clean.set_index("simulator_id", inplace=True)
        self.master_components.update(component_gdf_clean)
        self.master_components.reset_index(inplace=True)
        component_gdf_clean.reset_index(inplace=True)
        
        

    def on_change(self, source, property_name, old_value, new_value):
        """
        Responds to changes in the simulator mode.

        Inputs:
            source (Simulator): The simulator that changed mode.
            property_name (str): The name of the property that changed.
            old_value (Mode): The old value of the property.
            new_value (Mode): The new value of the property.
        """
        if property_name == Simulator.PROPERTY_MODE and new_value == Mode.EXECUTING:
            logger.info(
                "Switched to EXECUTING mode. Counter and master components list initialized."
            )
        if property_name == Simulator.PROPERTY_MODE and new_value == Mode.TERMINATING:
            logger.info(
                "Switched to TERMINATING mode. Shutting down the application and saving data."
            )


def main():
    # Load config
    config = ConnectionConfig(yaml_file="sos.yaml")

    # Define the simulation parameters
    NAME = "appender"

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
        config.rc.simulation_configuration.execution_parameters.general.prefix,
        config,
        True,
    )
    app.add_message_callback("planner", "selected_cells", environment.on_planner)
    app.add_message_callback("simulator", "simulator_daily", environment.on_simulator)

    while True:
        pass


if __name__ == "__main__":
    main()
