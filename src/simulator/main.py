# Main Execution Script
# Author: Divya Ramachandran

import logging
from datetime import datetime, timedelta, timezone
from nost_tools.application_utils import ShutDownObserver
from nost_tools.configuration import ConnectionConfig
from nost_tools.managed_application import ManagedApplication
from nost_tools.observer import ScenarioTimeIntervalCallback
from sos_sim.entity import Collect_Observations
from sos_sim.function import Snowglobe_constellation, write_back_to_appender

# from sos_sim.observers import ScenarioTimeIntervalCallback
start_time = datetime(2019, 3, 1, tzinfo=timezone.utc)

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""""
NOST integration
""" ""
#  Load config
config = ConnectionConfig(yaml_file="sos.yaml")

# Define the simulation parameters
NAME = "simulator"

# create the managed application
app = ManagedApplication(NAME)
simulator = app.simulator

# add a shutdown observer to shut down after a single test case
app.simulator.add_observer(ShutDownObserver(app))

# start up the application on PREFIX, publish time status every 10 seconds of wallclock time
app.start_up(
    config.rc.simulation_configuration.execution_parameters.general.prefix, config
)


def log_observation(observation):
    """
    Log observation collection.
    """
    logger.info(
        "Request %s collected by %s at %s",
        observation["point_id"],
        "completed",
        observation["epoch"],
        observation["satellite"],
    )

time_step_callback = timedelta(days=1)  # time step for callback
# time_scale_factor = 60  # 5 seconds wallclock for 5 minutes scenario
# view_time_step = timedelta(seconds=2)  # time step for ground track

# app = Application("simulator")

# Initial Requests
# master = read_master_file()
# request_data = filter_requests(master)

# Add Collect_Observations entity
entity = Collect_Observations(
    constellation=Snowglobe_constellation(start_time), requests=[], application=app
)

# app.add_message_callback("appender", "topic", entity.message_received_from_appender)

simulator.add_entity(entity)
# entity.add_observer(
#         PropertyChangeCallback(Collect_Observations.PROPERTY_OBSERVATION, log_observation)
#     )

# Add Observers
entity.add_observer(
    ScenarioTimeIntervalCallback(write_back_to_appender, time_step_callback)
)

# # initialize the simulator
# simulator.initialize(start, None, time_scale_factor)

# # execute the simulator
# simulator.execute(start, duration, time_step, None, time_scale_factor)

app.add_message_callback("appender", "master", entity.message_received_from_appender)

