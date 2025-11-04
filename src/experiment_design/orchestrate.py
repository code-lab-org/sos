import logging
import time
import pandas as pd
import subprocess

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
        self.shutdown_received = True
        return 

        # Seeing content of body
        # logger.info(f"Body content: {body}")

    def on_change(self, ch, method, properties, body):
        logger.info("Received change message.")
        logger.info(f"Change content: {body}")

    def send_execute_command(self, command: str):
        logger.info(f"Sending execute command: {command}")
        subprocess.run("docker-compose up -d", shell=True, check=True, capture_output=True, text=True)
        # Add logic to send execute command

    def wait_for_shutdown(self):
        # global shutdown_received
        # shutdown_received = self.shutdown_received
        print("Waiting for shutdown...")
        while not self.shutdown_received:
            time.sleep(3)  # Sleep for 3 seconds before checking again
            logger.info("Still waiting for shutdown signal...")
        logger.info("Sleep complete, shutdown signal received.")
        self.shutdown_received = False
        logger.info("Setting docker compose down in nost environment.")
        subprocess.run("docker-compose down", shell=True, check=True, capture_output=True, text=True)
        logger.info("Sleeping for 30 seconds to ensure proper shutdown.")
        time.sleep(30)
        print("Proceeding to next iteration.")

    # def update_yaml_config(self, config, row):
    #     logger.info("Opening yaml file")
    #     # Add logic to update YAML configuration
    #     config.yaml_config.execution.manager.time_scale_factor = int(row["time scale factor"])
    #     config.yaml_config.execution.managed_applications["planner"].time_scale_factor = int(row["time scale factor"])
    #     config.yaml_config.execution.managed_applications["simulator"].time_scale_factor = int(row["time scale factor"])
    #     config.yaml_config.execution.managed_applications["appender"].time_scale_factor = int(row["time scale factor"])
    #     config.yaml_config.execution.managed_applications["planner"].configuration_parameters["budget"] = [int(row["budget"])]
    #     config.yaml_config.execution.managed_applications["simulator"].configuration_parameters["observation_interval"] = [int(row["observation interval"])]
    #     config.yaml_config.execution.managed_applications["simulator"].configuration_parameters["constellation_capacity"] = [int(row["constellation capacity"])]
    #     config.yaml_config.execution.managed_applications["appender"].configuration_parameters["expiration_time"] = [int(row["expiration"])]
    #     # Save the updated config back to the file
    #     with open(config.yaml_file, "w") as f:
    #         yaml.safe_dump(
    #             config.yaml_config.model_dump(),
    #             f,
    #             sort_keys=False,
    #             indent=2  # (2 is consistent with your current YAML format)
    #         )
    #     logger.info("YAML configuration updated with new parameters.")

    def update_yaml_config(self, config, row):
        logger.info("Opening YAML file: %s", config.yaml_file)

        # Load YAML as a plain Python dict
        with open(config.yaml_file, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        # --- Update values directly ---
        # Manager & application time scale factor
        time_scale = int(row["time scale factor"])
        yaml_data["execution"]["manager"]["time_scale_factor"] = time_scale
        yaml_data["execution"]["managed_applications"]["planner"]["time_scale_factor"] = time_scale
        yaml_data["execution"]["managed_applications"]["simulator"]["time_scale_factor"] = time_scale
        yaml_data["execution"]["managed_applications"]["appender"]["time_scale_factor"] = time_scale

        # Configuration parameters
        yaml_data["execution"]["managed_applications"]["planner"]["configuration_parameters"]["budget"] = [int(row["budget"])]
        yaml_data["execution"]["managed_applications"]["simulator"]["configuration_parameters"]["observation_interval"] = [int(row["observation interval"])]
        yaml_data["execution"]["managed_applications"]["simulator"]["configuration_parameters"]["constellation_capacity"] = [int(row["constellation capacity"])]
        yaml_data["execution"]["managed_applications"]["appender"]["configuration_parameters"]["expiration_time"] = [int(row["expiration"])]

        # --- Save updated YAML ---
        with open(config.yaml_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(yaml_data, f, sort_keys=False, indent=2)

        logger.info("YAML configuration updated with new parameters. Sleeping for 10 seconds.")
        time.sleep(10)

def main():
    logger.info("Entering main function")
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
    logger.info("comfig yaml content: %s", config.yaml_config)

    # Load CSV
    df = pd.read_csv("src/experiment_design/experiment_run_data.csv")

    # # print(df)
    app.add_message_callback("manager", "start", environment.on_start)
    app.add_message_callback("manager", "stop", environment.on_stop)
    logger.info("Exiting main function")

    for idx, row in df.iterrows():
        if idx == 0:
            subprocess.run("docker-compose down", shell=True, check=True, capture_output=True, text=True)
            logger.info("Sleeping for 30 seconds to ensure proper shutdown.")
            time.sleep(30)
        logger.info("Processing row: %s", row.to_dict())
        environment.update_yaml_config(config, row)
        environment.send_execute_command(f"Process data for {row['Run']}")
        environment.wait_for_shutdown()
        logger.info("Completed processing for row: %s", row.to_dict())

if __name__ == "__main__":
    main()
