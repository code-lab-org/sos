import json
import logging
import os
import sys
from datetime import timedelta,datetime
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

    def __init__(self, app, set_expiration = False, expiration_time=None, enable_uploads=None):
        self.app = app
        self.counter = 0
        self.master_components = []
        self.master_gdf = gpd.GeoDataFrame()
        self.visualize_selected = False  # True
        self.set_expiration = set_expiration
        self.expiration_time = expiration_time

        # Flag to control S3 uploads - check environment variable if not explicitly set
        if enable_uploads is None:
            self.enable_uploads = os.environ.get("ENABLE_UPLOADS", "true").lower() in (
                "true",
                "1",
                "yes",
            )
        else:
            self.enable_uploads = enable_uploads

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
        gdf["simulator_simulation_status"] = pd.Series(dtype="string")
        gdf["simulator_completion_date"] = pd.NaT 
        

        if self.set_expiration:
            logger.info(f"Setting expiration date with expiration time of %s days", self.expiration_time)
            gdf["simulator_expiration_date"] = pd.to_datetime(
                gdf["planner_time"], utc=True
            ) + timedelta(days=int(self.expiration_time))
        else:            
            logger.info("Expiration time not set setting expiration date as infinite (future date)")
            gdf["simulator_expiration_date"] = pd.Timestamp.max.tz_localize("UTC")

        gdf["simulator_expiration_status"] = pd.Series(dtype="string")
        gdf["simulator_satellite"] = pd.Series(dtype="string")
        gdf["simulator_polygon_groundtrack"] = np.nan  # None
        gdf["planner_latitude"] = gdf["planner_centroid"].y
        gdf["planner_longitude"] = gdf["planner_centroid"].x
        gdf["planner_centroid"] = gdf["planner_centroid"].to_wkt()
        
        # current_sim_time = self.app.simulator._time  # Must be datetime
        current_sim_time = self.app.simulator.get_time()
        if isinstance(current_sim_time, (int, float)):
            current_sim_time = datetime.fromtimestamp(current_sim_time)
        gdf["simulator_expiration_status"] = np.where(
            gdf["simulator_expiration_date"].dt.date < current_sim_time.date(),
            "expired",
            "active"
        )       
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
                "simulator_expiration_status",
                "simulator_simulation_status",
                "simulator_completion_date",
                "simulator_satellite",
                "simulator_polygon_groundtrack",
                "geometry",
            ]
        ]
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
        # Reproject to a CRS suitable for Missouri River Basin (Conus Albers)
        component_gdf_proj = component_gdf.to_crs(epsg=5070)
        # Compute centroid accurately in projected coordinates
        component_gdf["centroid"] = component_gdf_proj.centroid.to_crs(epsg=4326)
        # component_gdf["centroid"] = component_gdf.centroid
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
        k = gpd.GeoDataFrame.from_features(
            json.loads(data.vector_layer)["features"], crs="EPSG:4326"
        )
        logger.info("Conversion to GeoDataFrame completed")
        return k



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
        self.master_components.append(component_gdf)
        self.counter += len(component_gdf)
        # min_value = component_gdf["simulator_id"].min()
        # max_value = component_gdf["simulator_id"].max()
        self.master_gdf = pd.concat(self.master_components, ignore_index=True)
        self.remove_duplicates()
        self.master_gdf = self.master_gdf.sort_values(by="simulator_id").reset_index(drop=True)
        date = self.app.simulator.get_time()
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

        # Upload to S3 only if uploads are enabled
        if self.enable_uploads:
            logger.info("Uploading file to S3: %s", output_file)
            s3.upload_file(
                Bucket="snow-observing-systems",
                Key=output_file,
                Filename=output_file,
                Config=TransferConfig(use_threads=False),
            )
        else:
            logger.info("Upload skipped (uploads disabled): %s", output_file)

        self.master_gdf.to_file("outputs/master.geojson", driver="GeoJSON")
        logger.info("Master geosjon file created")

        # Filter based on expiration status, not equal to 'expired'
        # Filter if self.expiration is set else use master_gdf
        if self.set_expiration:
            filtered_gdf = self.master_gdf[
                self.master_gdf["simulator_expiration_status"] != "expired"
            ]
        else:
            filtered_gdf = self.master_gdf

        logger.info("Filtered gdf based on expiration: %d", len(filtered_gdf))

        # selected_json_data = self.master_gdf.to_json()
        selected_json_data = filtered_gdf.to_json()
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
            logger.info("%s sent message. at %s", self.app.app_name, self.app.simulator.get_time())

    def on_simulator(self, ch, method, properties, body):
        """
        Responds to messages from simulator application(updates self.master_gdf with simulator data)

        Inputs:
            ch (Channel): The channel on which the message was received.
            method (Method): The method used to receive the message.
            properties (Properties): The properties of the message.
            body (bytes): The body of the message.

        """
        logger.info("entering appender _on_simulator")
        component_gdf = self.message_to_geojson(body)
        
        if component_gdf.empty:
            logger.warning("Received empty message from simulator. Skipping update.")
            return

        import time as time
        start_time = time.perf_counter()

        # Combine self.master_components into a single GeoDataFrame
        master_gdf_combined = pd.concat(self.master_components, ignore_index=True)
        master_gdf_combined.set_index("simulator_id", inplace=True)
        component_gdf.set_index("simulator_id", inplace=True)
        master_gdf_combined.update(component_gdf)
        master_gdf_combined.reset_index(inplace=True)
        component_gdf.reset_index(inplace=True)        

        self.master_components = [
            gpd.GeoDataFrame(
                group.sort_values("simulator_id"),  # sort within each group
                geometry="geometry",
                crs="EPSG:4326",
            ).reset_index(drop=True)
            for _, group in master_gdf_combined.groupby("planner_time")
        ]              
        end_time = time.perf_counter()
        logger.info("Time taken to update master_gdf_combined: %f seconds", end_time - start_time)

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
    config = ConnectionConfig(yaml_file="sos.yaml",app_name="appender")

    # create the managed application
    app = ManagedApplication(app_name="appender")

    # initialize the Environment object class
    environment = Environment(app,
                              set_expiration = config.rc.application_configuration['set_expiration_time'][0],
                              expiration_time = config.rc.application_configuration['expiration_time'][0]
                                )

    app.simulator.add_observer(environment)
    # add a shutdown observer to shut down after a single test case
    app.simulator.add_observer(ShutDownObserver(app))
    # start up the application on PREFIX, publish time status every 10 seconds of wallclock time
    app.start_up(
        config.rc.simulation_configuration.execution_parameters.general.prefix,
        config,
        True,
    )

    # Add a message callback to handle messages from the planner
    app.add_message_callback("planner", "selected_cells", environment.on_planner)
    app.add_message_callback("simulator", "simulator_daily", environment.on_simulator)

    while True:
        pass


if __name__ == "__main__":
    main()
