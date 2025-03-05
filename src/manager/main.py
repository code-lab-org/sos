import logging

from nost_tools.application_utils import ShutDownObserver
from nost_tools.config import ConnectionConfig
from nost_tools.manager import Manager

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Load config
    config = ConnectionConfig(yaml_file="sos.yaml")

    # create the manager application from the template in the tools library
    manager = Manager()

    # add a shutdown observer to shut down after a single test case
    manager.simulator.add_observer(ShutDownObserver(manager))

    # start up the manager on PREFIX from config file
    manager.start_up(
        config.rc.simulation_configuration.execution_parameters.general.prefix,
        config,
        # True,
    )

    manager.execute_test_plan(
        # datetime(2024, 1, 7, tzinfo=timezone.utc),  # scenario start datetime
        # datetime(2024, 1, 21, tzinfo=timezone.utc),  # scenario stop datetime
        # start_time=None,  # optionally specify a wallclock start datetime for synchronization
        # time_step=timedelta(seconds=1),  # wallclock time resolution for simulation
        # time_scale_factor=SCALE,  # initial scale between wallclock and scenario clock (e.g. if SCALE = 60.0 then  1 wallclock second = 1 scenario minute)
        # time_scale_updates=UPDATE,  # optionally schedule changes to the time_scale_factor at a specified scenario time
        # time_status_step=timedelta(seconds=1)
        # * SCALE,  # optional duration between time status 'heartbeat' messages
        # time_status_init=datetime(
        #     2024, 1, 7, tzinfo=timezone.utc
        # ),  # optional initial scenario datetime to start publishing time status 'heartbeat' messages
        # command_lead=timedelta(
        #     seconds=5
        # ),  # lead time before a scheduled update or stop command
        # required_apps=[
        #     "swe_change"
        # ],  # ["mod10c1", "snow"]#["snodas", "resolution"] #"swe_change", #['snow', 'resolution', 'constellation'], # list of required applications
    )
