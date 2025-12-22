import logging
import shutil
import time
import os
import pandas as pd
import subprocess
import json
import csv
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

    def on_change(self, ch, method, properties, body):
        logger.info("Received change message.")
        logger.info(f"Change content: {body}")

    def on_start(self, ch, method, properties, body):
        logger.info("Received start message. Starting application.")

    def wait_for_shutdown(self):
        # global shutdown_received
        # shutdown_received = self.shutdown_received
        print("Waiting for shutdown...")
        while not self.shutdown_received:
            time.sleep(5)  # Sleep for 5 seconds before checking again
            logger.info("Still waiting for shutdown signal...")
        logger.info("Sleep complete, shutdown signal received.")
        self.shutdown_received = False

    def on_stop(self, ch, method, properties, body):
        logger.info("Received stop message. Stopping application.")

def main():
    logger.info("Entering main function")
    config = ConnectionConfig(yaml_file="sos.yaml")
    # logger.info(f"Contents of config loaded: {config}")
    app = Application(app_name="test")
    environment = OrchestrateObserver(app)
    app.simulator.add_observer(environment)
    app.start_up(
    config.rc.simulation_configuration.execution_parameters.general.prefix,
    config,
    True,
    )
    app.add_message_callback("simulator", "start", environment.on_start)
    app.add_message_callback("manager", "stop", environment.on_stop)
    # app.add_message_callback("simulator", "simulator_end", environment.on_stop)

    logger.info("Exiting main function")

if __name__ == "__main__":
    main()

    
