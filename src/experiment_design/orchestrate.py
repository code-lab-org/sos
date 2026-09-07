import logging
import shutil
import time
import os
import pandas as pd
import geopandas as gpd
import subprocess
import json
import csv
from shapely import wkt
from nost_tools.application import Application
from nost_tools.configuration import ConnectionConfig
from nost_tools.observer import Observer
import yaml
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrchestrateObserver(Observer):
    def __init__(self, app):
        self.app = app
        self.shutdown_received = False

    def observe(self):
        logger.info("OrchestrateObserver is observing...")
        # Add observation logic here

    def on_start(self, ch, method, properties, body):
        logger.info("Received start message. Starting application.")

    def on_stop(self, ch, method, properties, body):
        logger.info("Received stop message. Stopping application.")
        time.sleep(30)  # Wait for 1.5 minutes before proceeding
        self.shutdown_received = True
        return
    
    def on_change(self, ch, method, properties, body):
        logger.info("Received change message.")
        logger.info(f"Change content: {body}")

    def send_execute_command(self, command: str):
        logger.info(f"Sending execute command: {command}")
        subprocess.run("docker compose up -d", shell=True, check=True, capture_output=True, text=True)
        # Add logic to send execute command

    def wait_for_shutdown(self):
        # global shutdown_received
        # shutdown_received = self.shutdown_received
        print("Waiting for shutdown...")
        while not self.shutdown_received:
            time.sleep(60)  # Sleep for 3 seconds before checking again
            logger.info("Still waiting for shutdown signal...")

        logger.info("Sleep complete, shutdown signal received.")
        self.shutdown_received = False
        logger.info("Setting docker compose down in nost environment.")
        subprocess.run("docker compose down", shell=True, check=True, capture_output=True, text=True)
        logger.info("Sleeping for 60 seconds to ensure proper shutdown.")
        time.sleep(60)
        print("Proceeding to next iteration.")

    def update_yaml_config(self, config, row):

        logger.info("Opening YAML file: %s", config.yaml_file)
        # Load YAML as a plain Python dict
        with open(config.yaml_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        # --- Update values directly ---
        # Manager & application time scale factor
        time_scale = int(row["time scale factor"])
        # time_scale = 96  # TEMPORARY OVERRIDE FOR TESTING
        yaml_data["execution"]["manager"]["time_scale_factor"] = time_scale
        yaml_data["execution"]["managed_applications"]["planner"]["time_scale_factor"] = time_scale
        yaml_data["execution"]["managed_applications"]["simulator"]["time_scale_factor"] = time_scale
        yaml_data["execution"]["managed_applications"]["appender"]["time_scale_factor"] = time_scale

        # Configuration parameters
        yaml_data["execution"]["managed_applications"]["planner"]["configuration_parameters"]["budget"] = [int(row["budget"])]
        yaml_data["execution"]["managed_applications"]["simulator"]["configuration_parameters"]["observation_interval"] = [int(row["observation interval"])]
        yaml_data["execution"]["managed_applications"]["simulator"]["configuration_parameters"]["constellation_capacity"] = [float(row["constellation capacity"])]
        yaml_data["execution"]["managed_applications"]["appender"]["configuration_parameters"]["expiration_time"] = [int(row["expiration"])]

        # --- Save updated YAML ---
        with open(config.yaml_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, sort_keys=False, indent=2)

        logger.info("YAML configuration updated with new parameters. Sleeping for 10 seconds.")
        time.sleep(5)



def main():

    def geom_key(g):
        if isinstance(g, str):
            g = wkt.loads(g)
        return g.wkt

    def add_access_flag(gdf, metrics_df, flag_col):
        gdf[flag_col] = gdf["simulator_id"].isin(set(metrics_df["point_id"]))

        remaining = ~gdf[flag_col]
        metric_polys = set(metrics_df["planner_geometry"].apply(geom_key))

        gdf.loc[remaining, flag_col] = (
            gdf.loc[remaining, "geometry"].apply(geom_key).isin(metric_polys)
        )

        return gdf 

    logger.info("Entering main function")

    # Option 1: Delete the USERNAME variable entirely from the process environment
    if "USERNAME" in os.environ:
        del os.environ["USERNAME"] 

    config = ConnectionConfig(yaml_file="sos.yaml")
    # logger.info(f"Contents of config loaded: {config}")
    app = Application(app_name="orchestrate")
    environment = OrchestrateObserver(app)
    app.simulator.add_observer(environment)

    app.start_up(
    config.rc.simulation_configuration.execution_parameters.general.prefix,
    config,
    True,
    )

    # logger.info("Config loaded: %s", config)
    logger.info("config yaml content: %s", config.yaml_config)

    # Load CSV
    df = pd.read_csv("src/experiment_design/experiment_run_data.csv")
    
    app.add_message_callback("manager", "start", environment.on_start)
    app.add_message_callback("manager", "stop", environment.on_stop)
    # app.add_message_callback("simulator", "simulator_end", environment.on_stop)
    logger.info("Exiting main function")

    # Ensure run_output directory exists
    run_output_base = "run_output"
    os.makedirs(run_output_base, exist_ok=True)

    for idx, row in df.iterrows():
        if idx == 0:
            subprocess.run("docker compose down", shell=True, check=True, capture_output=True, text=True)
            logger.info("Sleeping for 30 seconds to ensure proper shutdown.")
            time.sleep(30)

        logger.info("Processing row: %s", row.to_dict())
        # -------------------------------------------
        # RECLAIM OWNERSHIP OF CONTAINER-WRITTEN OUTPUT (containers run as root)
        # -------------------------------------------
        outputs_root = os.path.abspath("outputs")
        if os.path.exists(outputs_root):
            subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{outputs_root}:/data",
                    "ubuntu", "chown", "-R", f"{os.getuid()}:{os.getgid()}", "/data",
                ],
                check=True, capture_output=True, text=True,
            )
            logger.info("Reclaimed ownership of %s", outputs_root)

        # -------------------------------------------
        # DELETE SIMULATOR SUBFOLDERS BEFORE EXECUTION
        # -------------------------------------------
        simulator_root = os.path.join("outputs", "simulator")
        # simulator_root = "outputs"
        if os.path.exists(simulator_root):
            for item in os.listdir(simulator_root):
                item_path = os.path.join(simulator_root, item)
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    logger.info("Deleted subfolder: %s", item_path)
        else:
            logger.warning("Simulator folder not found at %s", simulator_root)

        environment.update_yaml_config(config, row)
        environment.send_execute_command(f"Process data for {row['Run']}")
        environment.wait_for_shutdown()
        logger.info("Completed processing for row: %s", row.to_dict())
        # Copy the outputs folder with run number
        # Copy the output folder
        src_folder = "outputs"
        dest_folder = os.path.join(run_output_base, f"run_{row['Run']}_output")

        if os.path.exists(src_folder):
            shutil.copytree(src_folder, dest_folder, dirs_exist_ok=True)
            logger.info("Copied %s to %s", src_folder, dest_folder)
        else:
            logger.warning("Source folder '%s' not found; skipping copy.", src_folder)

        # Creating summary for the runs
        geojson_path = os.path.join(dest_folder, "master.geojson")
        metrics_path = os.path.join(dest_folder, "metrics/geometrically_accessible_aggregated.csv")
        metrics_nolatency_path = os.path.join(dest_folder, "metrics/geometrically_accessible_aggregated_no_latency.csv") 
        lost_simulation_path = os.path.join(dest_folder, "metrics/lost_simulation_time.csv")
        csv_path = os.path.join(dest_folder, "simulation_config.csv")

        if os.path.exists(geojson_path) and os.path.exists(metrics_path) and os.path.exists(metrics_nolatency_path) and os.path.exists(lost_simulation_path):
            logger.info("Entering summary generation for run %s", row['Run'])
            with open(geojson_path, encoding="utf-8") as f:
                feats = json.load(f).get("features", [])
            total = len(feats)
            # Sum of all the 'planner_final_eta' values (total) for all features
            eta_sum_total = sum(
                f.get("properties", {}).get("planner_final_eta", 0)
                for f in feats if isinstance(f.get("properties", {}).get("planner_final_eta"), (int, float))
            )

            # Sum of all the 'planner_final_eta' values for features with 'simulator_simulation_status' equal to 'Completed'
            completed = [f for f in feats if f.get("properties", {}).get("simulator_simulation_status") == "Completed"]
            eta_sum = sum(
                f.get("properties", {}).get("planner_final_eta", 0)
                for f in completed if isinstance(f.get("properties", {}).get("planner_final_eta"), (int, float))
            )

            # Count of expired requests "expiration_status" equal to "Expired"
            expired = [f for f in feats if f.get("properties", {}).get("simulator_expiration_status") == "expired"]
            expired_count = len(expired)
            # Sum of planner_final_eta for expired requests
            eta_sum_expired = sum(
                f.get("properties", {}).get("planner_final_eta", 0)
                for f in expired if isinstance(f.get("properties", {}).get("planner_final_eta"), (int, float))
            )

            # Count and fraction of values where 'simulation_simulation_status' is "pending" and not expired
            pending = [
                f for f in feats
                if f.get("properties", {}).get("simulator_simulation_status") == "pending"
                and f.get("properties", {}).get("simulator_expiration_status") != "expired"
            ]
            pending_count = len(pending)
            fraction_pending = pending_count / total if total > 0 else 0

            
            # Calculate fraction of values
            fraction_completed = len(completed) / total if total > 0 else 0
            fraction_expired = expired_count / total if total > 0 else 0
            fraction_eta_completed = eta_sum / eta_sum_total if eta_sum_total > 0 else 0
            fraction_eta_expired = eta_sum_expired / eta_sum_total if eta_sum_total > 0 else 0
            avg_eta = eta_sum / len(completed) if completed else 0
            logger.info("Total=%d Completed=%d ETA_Sum=%.2f", total, len(completed), eta_sum)

            # Join geojson_path and metrics_path to get 'first_access_time' for each 'point_id'(in metrics_df) and calculate average time to first access wrt to 'planner_time' in (geojson_path(gdf))
            gdf = gpd.read_file(geojson_path)

            # Metrics DataFrame to count unique 'point_id' values           
            metrics_df = pd.read_csv(metrics_path)
            metrics_nolatency_df = pd.read_csv(metrics_nolatency_path)

            # Add access flag to indicate whether each point in the GeoDataFrame has been accessed according to the metrics DataFrame
            gdf = add_access_flag(gdf, metrics_df, "accessible_with_latency")
            gdf = add_access_flag(gdf, metrics_nolatency_df, "accessible_without_latency")
    
            # Count the number of unique 'point_id' values (calculates whether they are accessible)
            unique_point_ids = gdf["accessible_with_latency"].sum()
            unique_point_ids_nolatency = gdf["accessible_without_latency"].sum()
            difference_unique_points = unique_point_ids_nolatency - unique_point_ids
            fraction_unique_points = unique_point_ids / total if total > 0 else 0
            fraction_never_accessible = 1 - fraction_unique_points if total > 0 else 0


            # Compute time to completion within the master file
            gdf["planner_time"] = pd.to_datetime(gdf["planner_time"], utc=True, errors="coerce")
            gdf["simulator_completion_date"] = pd.to_datetime(gdf["simulator_completion_date"], utc=True, errors="coerce")
            gdf["time_to_completion"] = gdf["simulator_completion_date"] - gdf["planner_time"]
            gdf["time_to_completion_hours"] = gdf["time_to_completion"].dt.total_seconds() / 3600

            # Computing aggregate metrics for all requests within the entire simulation run
            avg_completion_hours = gdf["time_to_completion_hours"].mean()
            median_completion_hours = gdf["time_to_completion_hours"].median()

            # Computing Time to first access
            gdf = gdf.rename(columns={"simulator_id": "point_id"})  # Ensure the column names match for merging
            logger.info(" Columns in both the GeoDataFrame and metrics DataFrame: %s, %s", gdf.columns.tolist(), metrics_df.columns.tolist())
            # Getting the first access time for each point_id from metrics_df(with latency) and merging it with gdf(master.geojson)
            merged_df = pd.merge(gdf, metrics_df[["point_id", "first_access_time"]], on="point_id", how="left")

            merged_df["planner_time"] = pd.to_datetime(merged_df["planner_time"], utc=True, errors="coerce")
            merged_df["first_access_time"] = pd.to_datetime(merged_df["first_access_time"], utc=True, errors="coerce")
            # Compute time from 'planner_time' to 'first_access_time' for each point_id
            merged_df["time_to_first_access"] = merged_df["first_access_time"] - merged_df["planner_time"]
            merged_df["time_to_first_access_hours"] = merged_df["time_to_first_access"].dt.total_seconds() / 3600

            merged_output_path = os.path.join(
                dest_folder,
                "metrics/master_with_geometric_access_check.csv",
            )

            merged_df.to_csv(merged_output_path, index=False)

            logger.info(
                "Saved master geometric access check file to %s",
                merged_output_path,
            )

            avg_hours = merged_df["time_to_first_access_hours"].mean()
            median_hours = merged_df["time_to_first_access_hours"].median()

            # Lost simulation time metrics
            lost_simulation_df = pd.read_csv(lost_simulation_path)
            # Average "hours_lost" for all records
            avg_hours_lost = lost_simulation_df["hours_lost"].mean()

            # Acquisition efficiency (completed/accessible)
            acquisition_efficiency = len(completed) / unique_point_ids if unique_point_ids > 0 else 0

            if os.path.exists(csv_path):
                logger.info("CSV path exists: %s. Appending summary rows.", csv_path)
                with open(csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["summary", "total_records", total])
                    writer.writerow(["summary", "completed_records", len(completed)])
                    writer.writerow(["summary", "fraction_completed", fraction_completed])
                    writer.writerow(["summary", "eta_sum", eta_sum])
                    writer.writerow(["summary", "eta_avg", avg_eta])
                    writer.writerow(["summary", "geometrically_accessible_points", unique_point_ids])
                    writer.writerow(["summary", "access_fraction", fraction_unique_points])
                    writer.writerow(["summary", "fraction_never_accessible", fraction_never_accessible])                    
                    writer.writerow(["summary", "acquisition_efficiency", acquisition_efficiency])
                    writer.writerow(["summary", "geometrically_accessible_points_nolatency", unique_point_ids_nolatency])
                    writer.writerow(["summary", "lost_access_latency", difference_unique_points])
                    writer.writerow(["summary", "expired_records", expired_count])
                    writer.writerow(["summary", "fraction_expired", fraction_expired])
                    writer.writerow(["summary", "pending_records", pending_count])
                    writer.writerow(["summary", "fraction_pending", fraction_pending])
                    writer.writerow(["summary", "reward_weighted_coverage", fraction_eta_completed])
                    writer.writerow(["summary", "reward_weighted_expired", fraction_eta_expired])
                    writer.writerow(["summary", "avg_time_to_first_access_hours", avg_hours])
                    writer.writerow(["summary", "median_time_to_first_access_hours", median_hours])
                    writer.writerow(["summary", "avg_time_to_completion_hours", avg_completion_hours])
                    writer.writerow(["summary", "median_time_to_completion_hours", median_completion_hours])
                    writer.writerow(["summary", "avg_hours_lost", avg_hours_lost])
                logger.info("Appended summary rows to %s", csv_path)
                logger.info("Execution completed for run %s. Summary metrics appended to CSV.", row['Run'])
            else:
                logger.warning("CSV '%s' not found; skipping append.", csv_path)
        else:
            logger.warning("GeoJSON '%s' not found; skipping processing.", geojson_path)


if __name__ == "__main__":
    main()