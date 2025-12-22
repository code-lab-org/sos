import logging
import pandas as pd
from datetime import datetime, timedelta
import os

from nost_tools.application_utils import ShutDownObserver
from nost_tools.configuration import ConnectionConfig
from nost_tools.manager import Manager

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Load config
    config = ConnectionConfig(yaml_file="sos.yaml")
    planner_config = ConnectionConfig(yaml_file="sos.yaml",app_name="planner")
    appender_config = ConnectionConfig(yaml_file="sos.yaml",app_name="appender")
    simulator_config = ConnectionConfig(yaml_file="sos.yaml",app_name="simulator")

    manager_exec = config.rc.simulation_configuration.execution_parameters.manager
    # logger.info(f"Manager Execution Parameters: {manager_exec.sim_start_time}")

    # Sample values 
    sim_start_time_str = manager_exec.sim_start_time
    sim_stop_time_str = manager_exec.sim_stop_time
    time_step_str = manager_exec.time_step
    time_scale_factor = manager_exec.time_scale_factor

    # Convert strings to datetime
    sim_start = sim_start_time_str
    sim_stop = sim_stop_time_str
    # time_step = timedelta(seconds=5)  # from "0:00:01"
    sim_duration = sim_stop - sim_start

    # Application values
    constellation_capacity = simulator_config.rc.application_configuration['constellation_capacity'][0]
    observation_window = simulator_config.rc.application_configuration['observation_interval'][0]
    set_expiration = appender_config.rc.application_configuration['set_expiration_time'][0]
    expiration_time = appender_config.rc.application_configuration['expiration_time'][0]
    budget = planner_config.rc.application_configuration['budget'][0]

    config_rows = [
    # Manager
    {'component': 'manager', 'parameter': 'sim_start_time', 'value': sim_start_time_str},
    {'component': 'manager', 'parameter': 'sim_stop_time', 'value': sim_stop_time_str},
    {'component': 'manager', 'parameter': 'sim_duration_days', 'value': sim_duration.days},
    {'component': 'manager', 'parameter': 'time_step', 'value': time_step_str},
    {'component': 'manager', 'parameter': 'time_scale_factor', 'value': time_scale_factor},

    # Planner
    {'component': 'planner', 'parameter': 'budget', 'value': budget},

    # Appender
    {'component': 'appender', 'parameter': 'set_expiration_time', 'value': set_expiration},
    {'component': 'appender', 'parameter': 'expiration_time', 'value': expiration_time},

    # Simulator
    {'component': 'simulator', 'parameter': 'constellation_capacity', 'value': constellation_capacity},
    {'component': 'simulator', 'parameter': 'observation_interval', 'value': observation_window},
    ]

    # Create the DataFrame
    df = pd.DataFrame(config_rows)

    # Ensure the outputs directory exists
    OUTPUT_DIRECTORY = "outputs"
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    # Save DataFrame to CSV inside outputs folder
    output_path = os.path.join(OUTPUT_DIRECTORY, "simulation_config.csv")
    df.to_csv(output_path, index=False)

    print(f"Saved simulation config to: {output_path}")

    # # Save to CSV if needed
    # df.to_csv("outputs/simulation_config.csv", index=False)

    # create the manager application from the template in the tools library
    manager = Manager()

    # add a shutdown observer to shut down after a single test case
    manager.simulator.add_observer(ShutDownObserver(manager))

    # logger.info("Starting Manager Application")
    # logger.info(f"Simulation time step : {config.rc.simulation_configuration.execution_parameters.manager.time_step}")
    # logger.info(f"time step :{time_step}")
    # logger.info(f"Simulation time step : {config.rc.simulation_configuration.execution_parameters.manager.time_step}")
    # logger.info(f"self.app.")
   

    # start up the manager on PREFIX from config file
    manager.start_up(
        config.rc.simulation_configuration.execution_parameters.general.prefix,
        config,
        True,
    )

    manager.execute_test_plan()
