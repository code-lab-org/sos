# Main Execution Script
# Author: Divya Ramachandran

from datetime import datetime, timedelta, timezone
import logging
from typing import List
from nost_tools import Simulator, Application
from pydantic import TypeAdapter
import pandas as pd
from tatc.schemas import Satellite as TATC_Satellite
import geopandas as gpd
from sos_sim.observers import ScenarioTimeIntervalCallback, PropertyChangeCallback
#  import ScenarioTimeIntervalCallback, PropertyChangeCallback
# from sos_sim.schemas import Observation, Request
# from tatc.analysis import collect_ground_track
# from tatc.analysis import compute_ground_track
# from tatc.schemas import PointedInstrument, WalkerConstellation, SunSynchronousOrbit
# from tatc.analysis import collect_multi_observations
# from tatc.utils import swath_width_to_field_of_regard, swath_width_to_field_of_view
# from tatc.analysis import collect_multi_observations

from tatc.schemas import Satellite
from tatc.schemas import Point
from sos_sim.function import (
    read_master_file,
    # compute_opportunity,
    # update_requests,
    Snowglobe_constellation,
    # compute_ground_track_and_format,
    # filter_requests,
    write_back_to_appender,
)
from sos_sim.entity import Collect_Observations

# # Function to save observations
# observations_list = []    

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# # Logging values
# def log_observations(observation):
    
#     """
#     Function to log_observations
#     """
#     observation_data = {
#     'simulator_id': observation['point_id'],  # Access the 'uid' column from the GeoSeries
#     'status': 'completed',
#     'time': observation['epoch'],
#     'satellite': observation['satellite'],
#     'groundtrack': observation['groundtrack']          
#     }

#     observations_list.append(observation_data)    
#     # Logging the data
#     logger.info(
#         "Request %s collected by %s at %s",
#         observation['point_id'],  # Access the 'uid' column from the GeoSeries
#         'completed',
#         observation['epoch'],
#         observation['satellite']
#     ) 

def log_observation(observation):
    """
    Log observation collection.
    """
    logger.info(
        "Request %s collected by %s at %s",
        observation['point_id'], 
        'completed',
        observation['epoch'],
        observation['satellite'])

# configure scenario
start = datetime(2025, 1, 16, tzinfo=timezone.utc)  # nost simulation start
duration = timedelta(hours=1)  # nost simulation duration
time_step = timedelta(minutes=1)  # nost simulation time step
time_step_callback = timedelta(days=1)  # time step for callback
time_scale_factor = 60  # 5 seconds wallclock for 5 minutes scenario
view_time_step = timedelta(seconds=2)  # time step for ground track

app = Application("simulator")

simulator = app.simulator

# Initial Requests
master = read_master_file()
# request_data = filter_requests(master)

# Add Collect_Observations entity
entity = Collect_Observations(
    constellation=Snowglobe_constellation(start), 
    requests=read_master_file(), 
    application=app
)

# app.add_message_callback("appender", "topic", entity.message_received_from_appender)

simulator.add_entity(entity)
entity.add_observer(
        PropertyChangeCallback(Collect_Observations.PROPERTY_OBSERVATION, log_observation)
    )

# Add Observers
entity.add_observer(
    ScenarioTimeIntervalCallback(write_back_to_appender, time_step_callback)
)

# initialize the simulator
simulator.initialize(start, None, time_scale_factor)

# execute the simulator
simulator.execute(start, duration, time_step, None, time_scale_factor)
