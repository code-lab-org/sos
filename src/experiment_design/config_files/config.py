import os
from skyfield.api import load

PREFIX = os.getenv("PREFIX", "sos")
NAME = "swe_change"
LOG = f"\x1b[1m[\x1b[34m{NAME}\x1b[37m]\x1b[0m"
HEADER = {
    "name": NAME,
    "description": "Satellites broadcast location on greenfield/satellite/location, and detect and report events on greenfield/satellite/detected and greenfield/satellite/reported. Subscribes to greenfield/fire/location.",
}