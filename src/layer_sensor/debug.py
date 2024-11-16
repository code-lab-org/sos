from dotenv import load_dotenv
import os
import geopandas as gpd
from shapely import Polygon

# Load environment variables from the .env file
load_dotenv('/Users/hbanafsh/Documents/GitHub/Code-lab/src/a.env')

# Retrieve the path_shp variable
path_shp = os.getenv('path_shp')

# Print the output of path_shp
print(f"The value of path_shp is: {path_shp}")

# Use os.path.join to construct the full path to the shapefile
shapefile_path = os.path.join(path_shp, "WBD_10_HU2.shp")

# Print the constructed shapefile path for debugging
print(f"The full shapefile path is: {shapefile_path}")

# Read the shapefile using Geopandas
mo_basin = gpd.read_file(shapefile_path)

# Construct a geoseries with the exterior of the basin and WGS84 coordinates
mo_basin = gpd.GeoSeries(Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326")
