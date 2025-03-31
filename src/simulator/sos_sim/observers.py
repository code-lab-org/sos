"""
Provides simulation observers.
"""

from collections.abc import Callable
from datetime import datetime, timedelta

from nost_tools import Mode, Observer, Simulator


class PropertyChangeCallback(Observer):
    """
    Triggers a provided callback basedwhen a named property changes.
    """

    def __init__(self, property_name: str, callback: Callable[[object, object], None]):
        self.callback = callback
        self.property_name = property_name

    def on_change(
        self, source: object, property_name: str, old_value: object, new_value: object
    ) -> None:
        if self.property_name == property_name:
            self.callback(source, new_value)


class ScenarioTimeIntervalCallback(Observer):
    """
    Triggers a provided callback at a fixed interval in scenario time.
    """

    def __init__(
        self, callback: Callable[[object, datetime], None], time_inteval: timedelta
    ):
        self.callback = callback
        self.time_interval = time_inteval
        self._next_time = None

    def on_change(
        self, source: object, property_name: str, old_value: object, new_value: object
    ):
        if property_name == source.PROPERTY_TIME:
            if self._next_time is None:
                self._next_time = old_value + self.time_interval
            while self._next_time <= new_value:
                self.callback(source, self._next_time)
                self._next_time = self._next_time + self.time_interval


class WallclockTimeIntervalCallback(Observer):
    """
    Triggers a provided callback at a fixed interval in wallclock time.
    """

    def __init__(
        self,
        simulator: Simulator,
        callback: Callable[[datetime], None],
        time_inteval: timedelta,
        time_init: timedelta = None,
    ):
        self.simulator = simulator
        self.callback = callback
        self.time_interval = time_inteval
        self.time_init = time_init
        self._next_time = None

    def on_change(
        self, source: object, property_name: str, old_value: object, new_value: object
    ):
        if property_name == Simulator.PROPERTY_MODE and new_value == Mode.INITIALIZED:
            self._next_time = self.time_init
        elif property_name == Simulator.PROPERTY_TIME:
            wallclock_time = self.simulator.get_wallclock_time()
            if self._next_time is None:
                self._next_time = wallclock_time + self.time_interval
            while self._next_time <= wallclock_time:
                self.callback(self._next_time)
                self._next_time = self._next_time + self.time_interval
