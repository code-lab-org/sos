# Main Execution Script
# Author: Divya Ramachandran

import logging
from datetime import timedelta

from nost_tools.application_utils import ShutDownObserver
from nost_tools.configuration import ConnectionConfig
from nost_tools.managed_application import ManagedApplication
from nost_tools.observer import ScenarioTimeIntervalCallback
from sos_sim.entity import Collect_Observations, RandomValueGenerator
from sos_sim.function import Snowglobe_constellation, write_back_to_appender

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    #  Load config
    config = ConnectionConfig(yaml_file="sos.yaml",app_name="simulator")

    # create the managed application
    app = ManagedApplication(app_name="simulator")    

    # Add Collect_Observations entity
    entity = Collect_Observations(
        constellation=Snowglobe_constellation(
            config.rc.simulation_configuration.execution_parameters.manager.sim_start_time
        ),
        requests=[],
        application=app,        
        const_capacity=config.rc.application_configuration["constellation_capacity"][0],
        time_interval=config.rc.application_configuration["observation_interval"][0],
        enable_uploads=None,  # Will check ENABLE_UPLOADS environment variable
    )    

    # Add observer classes to constellation's object class
    entity.add_observer(RandomValueGenerator(app))
    app.simulator.add_entity(entity)

    # Add a ScenarioTimeIntervalCallback to write back to the appender every day
    entity.add_observer(
        ScenarioTimeIntervalCallback(write_back_to_appender, timedelta(days=1))
    )

    # add a shutdown observer to shut down after a single test case
    app.simulator.add_observer(ShutDownObserver(app))

    # start up the application on PREFIX, publish time status every 10 seconds of wallclock time
    app.start_up(
        config.rc.simulation_configuration.execution_parameters.general.prefix,
        config,
        True,
    )

    # Add a message callback to handle messages from the appender
    app.add_message_callback(
        "appender", "master", entity.message_received_from_appender
    )

    # app.simulator.add_entity(
    #     SatelliteVisualization(
    #         constellation=Snowglobe_constellation(start_time), application=app
    #     )
    # )

    while True:
        pass


if __name__ == "__main__":
    main()
