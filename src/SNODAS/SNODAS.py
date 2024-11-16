from datetime import datetime
import gzip
import os
import requests
import shutil
import tarfile
import tempfile
from osgeo import gdal 
import pandas as pd

# define the date range over which to prepare data
dates = pd.date_range(datetime(2024,1,20), datetime(2024,1,21))

# define the local directory in which to work
snodas_dir = "SNODAS"
# ensure the directory exists
if not os.path.exists(snodas_dir):
    os.makedirs(snodas_dir)

# iterate over each date
for date in dates:
    # prepare the SWE file label for this date
    file_label = f"us_ssmv11034tS__T0001TTNATS{date.strftime('%Y')}{date.strftime('%m')}{date.strftime('%d')}05HP001"
    # check if file already exists
    if os.path.isfile(os.path.join(snodas_dir, file_label + ".nc")):
        print("Skipping " + file_label)
        continue
    print("Processing " + file_label)
    # prepare the SNODAS directory label for this date
    dir_label = f"SNODAS_{date.strftime('%Y%m%d')}"
    # request the .tar file from NSIDC
    r = requests.get(
        "https://noaadata.apps.nsidc.org/NOAA/G02158/masked/" + 
        f"{date.strftime('%Y')}/{date.strftime('%m')}_{date.strftime('%b')}/" +
        dir_label + ".tar"
    )
    # create a temporary directory in which to do work
    with tempfile.TemporaryDirectory() as tmp_dir:
        # save the .tar file
        with open(os.path.join(tmp_dir, dir_label + ".tar"), "wb") as tar_file:
            tar_file.write(r.content)
        # open and extract the .tar file
        with tarfile.open(os.path.join(tmp_dir, dir_label + ".tar"), "r") as tar_file:
            tar_file.extractall(tmp_dir)
        # iterate through all extracted files
        for filename in os.listdir(tmp_dir):
            # check if the file matches the SWE file label
            if os.path.isfile(os.path.join(tmp_dir, filename)) and filename == file_label + ".dat.gz":
                # unzip the SWE .gz file
                with gzip.open(os.path.join(tmp_dir, file_label + ".dat.gz"), "rb") as gz_in:
                    with open(os.path.join(tmp_dir, file_label + ".dat"), "wb") as gz_out:
                        shutil.copyfileobj(gz_in, gz_out)
                # write the SWE .hdr file
                with open(os.path.join(tmp_dir, file_label + ".hdr"), "w") as hdr_file:
                    hdr_file.write(
                        "ENVI\n"
                        "samples = 6935\n" +
                        "lines = 3351\n" +
                        "bands = 1\n" +
                        "header offset = 0\n" + 
                        "file type = ENVI Standard\n" + 
                        "data type = 2\n" +
                        "interleave = bsq\n" +
                        "byte order = 1"
                    )
                # run the gdal translator using date-specific bounding box
                command = " ".join([
                    "gdal_translate",
                    "-of NetCDF",
                    "-a_srs EPSG:4326",
                    "-a_nodata -9999",
                    "-a_ullr -124.73375000000000 52.87458333333333 -66.94208333333333 24.94958333333333" 
                    if date < datetime(2013, 10, 1)
                    else "-a_ullr -124.73333333333333 52.87500000000000 -66.94166666666667 24.95000000000000",
                    os.path.join(tmp_dir, file_label + ".dat"),
                    os.path.join(snodas_dir, file_label + ".nc")
                ])
                if os.system(command) > 0: 
                    print(f"Error processing command `{command}`")

import rioxarray as rxr
import xarray as xr
from datetime import datetime
import pandas as pd
import os

dates = pd.date_range(datetime(2024,1,20), datetime(2024,1,21))
print("Writing snodas-merged.nc")
ds = xr.combine_by_coords(
    [
        rxr.open_rasterio(
            os.path.join(
                snodas_dir, 
                f"us_ssmv11034tS__T0001TTNATS{date.strftime('%Y')}{date.strftime('%m')}{date.strftime('%d')}05HP001.nc"
            )
        ).drop_vars("band").assign_coords(time=date).expand_dims(dim="time")
        for date in dates
    ], 
    combine_attrs="drop_conflicts"
).to_netcdf(os.path.join('/Users/hbanafsh/ASU Dropbox/Hadis Banafsheh/SOS Planning', "snodas-merged.nc"))
