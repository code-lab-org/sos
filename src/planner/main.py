import base64
import io
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.enums import Resampling
from boto3.s3.transfer import TransferConfig
from constellation_config_files.schemas import SWEChangeLayer, VectorLayer
from joblib import Parallel, delayed
from nost_tools.application_utils import ShutDownObserver
from nost_tools.configuration import ConnectionConfig
from nost_tools.managed_application import ManagedApplication
from nost_tools.observer import Observer
from PIL import Image
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value
from rasterio.features import geometry_mask
from scipy.interpolate import griddata
from shapely.geometry import Polygon, box
from tatc import utils
from tatc.analysis import compute_ground_track
from tatc.schemas import (
    PointedInstrument,
    Satellite,
    SunSynchronousOrbit,
    TwoLineElements,
    WalkerConstellation,
)
from tatc.utils import swath_width_to_field_of_regard, swath_width_to_field_of_view

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.sos_tools.aws_utils import AWSUtils
from src.sos_tools.data_utils import DataUtils
from scipy.special import expit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class Environment(Observer):
    """
    *The Environment object class inherits properties from the Observer object class in the NOS-T tools library*

    Attributes:
        app (:obj:`ManagedApplication`): An application containing a test-run namespace, a name and description for the app, client credentials, and simulation timing instructions
        grounds (:obj:`DataFrame`): DataFrame of ground station information including groundId (*int*), latitude-longitude location (:obj:`GeographicPosition`), min_elevation (*float*) angle constraints, and operational status (*bool*)
    """

    def __init__(self, app):
        self.app = app
        self.visualize_swe_change = True
        self.visualize_all_layers = False
        self.current_simulation_date = None
        self.output_directory = os.path.join("outputs", self.app.app_name)
        self.input_directory = os.path.join("inputs", self.app.app_name)
        self.data_utils = DataUtils()
        self.data_utils.create_directories(
            [
                self.output_directory,
                self.input_directory,
                os.path.join("inputs", "vector"),
            ]
        )
        self.parallel_compute = True

    def interpolate_dataset(
        self, dataset, variables_to_interpolate, lat_coords, lon_coords, time_coords
    ):
        """
        Optimized interpolation of dataset to a new grid.

        Args:
            dataset (xarray.Dataset): The dataset to interpolate
            variables_to_interpolate (list): The variables to interpolate
            lat_coords (np.ndarray): The latitude coordinates
            lon_coords (np.ndarray): The longitude coordinates
            time_coords (np.ndarray): The time coordinates

        Returns:
            interpolated_dataset (xarray.Dataset): The interpolated dataset
        """
        # Precompute meshgrid once
        xi = np.stack(np.meshgrid(lon_coords, lat_coords), axis=2).reshape(
            (lat_coords.size * lon_coords.size, 2)
        )

        # Get coordinates once
        lons = dataset.lon.values.flatten()
        lats = dataset.lat.values.flatten()
        points = list(zip(lons, lats))

        interpolated_vars = {}

        for var_name in variables_to_interpolate:
            values = dataset[var_name].values.flatten()

            # Filter out NaN values to improve interpolation performance
            valid_mask = ~np.isnan(values)
            if not np.all(valid_mask):
                valid_points = [points[i] for i in range(len(points)) if valid_mask[i]]
                valid_values = values[valid_mask]
                zi = griddata(
                    valid_points, valid_values, xi, method="linear", fill_value=np.nan
                )
            else:
                zi = griddata(points, values, xi, method="linear")

            interpolated_vars[var_name] = xr.DataArray(
                np.reshape(zi, (1, len(lat_coords), len(lon_coords))),
                coords={"time": time_coords, "y": lat_coords, "x": lon_coords},
                dims=["time", "y", "x"],
            ).rio.write_crs("EPSG:4326")

        return xr.Dataset(interpolated_vars)

    def generate_combined_dataset(self, dataset1, dataset2, mo_basin):
        """
        Generate a combined dataset by interpolating the two datasets to a new grid and
        clipping to the Missouri Basin. Optimized for performance with parallel processing.

        Args:
            dataset1 (xarray.Dataset): The first dataset
            dataset2 (xarray.Dataset): The second dataset
            mo_basin (GeoSeries): The Missouri Basin polygon

        Returns:
            output_file (str): The output file name
            clipped_dataset (xarray.Dataset): The clipped dataset
        """
        # Check if output file already exists
        last_date_ds1 = np.datetime_as_string(
            dataset1.time[-1].values, unit="D"
        ).replace("-", "")
        last_date_ds2 = np.datetime_as_string(
            dataset2.time[-1].values, unit="D"
        ).replace("-", "")
        last_date = max(last_date_ds1, last_date_ds2)
        output_file = os.path.join(
            self.current_simulation_date, f"LIS_dataset_{last_date}.nc"
        )

        if os.path.exists(output_file):
            logger.info(
                f"File {output_file} already exists. Reading the existing file."
            )
            clipped_dataset = xr.open_dataset(output_file)
            return output_file, clipped_dataset

        logger.info("Combining the two datasets.")
        start_time = time.time()
        time_coords_ds1 = np.array([dataset1.time[0].values])
        time_coords_ds2 = np.array([dataset2.time[0].values])

        # Use fewer grid points if resolution can be reduced
        lat_coords = np.linspace(37.024602, 49.739086, 29)
        lon_coords = np.linspace(-113.938141, -90.114221, 40)
        variables_to_interpolate = ["SWE_tavg", "AvgSurfT_tavg","Snowcover_tavg"]

        # Parallelize dataset interpolation
        logger.debug("Starting parallel interpolation of datasets")
        results = Parallel(
            n_jobs=-1 if self.parallel_compute else 1, backend="threading"
        )(
            delayed(self.interpolate_dataset)(
                ds, variables_to_interpolate, lat_coords, lon_coords, time_coords
            )
            for ds, time_coords in [
                (dataset1, time_coords_ds1),
                (dataset2, time_coords_ds2),
            ]
        )

        new_ds1, new_ds2 = results
        logger.debug("Parallel interpolation completed")

        # Combine datasets and clip to Missouri Basin
        combined_dataset = xr.concat([new_ds1, new_ds2], dim="time")
        mo_basin = mo_basin.to_crs("EPSG:4326")
        combined_dataset = combined_dataset.rio.write_crs("EPSG:4326")

        logger.debug("Clipping combined dataset to Missouri Basin")
        clipped_dataset = combined_dataset.rio.clip(
            mo_basin.geometry, all_touched=True, drop=True
        )

        # Remove grid_mapping attributes
        for var in clipped_dataset.data_vars:
            if "grid_mapping" in clipped_dataset[var].attrs:
                del clipped_dataset[var].attrs["grid_mapping"]

        # Save to NetCDF
        logger.debug(f"Saving combined dataset to {output_file}")
        clipped_dataset.to_netcdf(output_file)
        end_time = time.time()
        logger.info(
            f"Combining the two datasets successfully completed in {end_time - start_time:.2f} seconds."
        )

        return output_file, clipped_dataset    

# MODIFIED BY DIVYA

    def generate_combined_dataset_resolution(self, dataset1, dataset2, mo_basin):
        """
        Customized for the resolution layer to interpolate to 1km grid
        Generate a combined dataset by interpolating the two datasets to a new grid and
        clipping to the Missouri Basin. Optimized for performance with parallel processing.

        Args:
            dataset1 (xarray.Dataset): The first dataset
            dataset2 (xarray.Dataset): The second dataset
            mo_basin (GeoSeries): The Missouri Basin polygon

        Returns:
            output_file (str): The output file name
            clipped_dataset (xarray.Dataset): The clipped dataset
        """
        # Check if output file already exists
        last_date_ds1 = np.datetime_as_string(
            dataset1.time[-1].values, unit="D"
        ).replace("-", "")
        last_date_ds2 = np.datetime_as_string(
            dataset2.time[-1].values, unit="D"
        ).replace("-", "")
        last_date = max(last_date_ds1, last_date_ds2)
        output_file = os.path.join(
            self.current_simulation_date, f"LIS_dataset_resolution_{last_date}.nc"
        )

        if os.path.exists(output_file):
            logger.info(
                f"File {output_file} already exists. Reading the existing file."
            )
            clipped_dataset = xr.open_dataset(output_file)
            return output_file, clipped_dataset

        logger.info("Combining the two datasets.")
        start_time = time.time()
        time_coords_ds1 = np.array([dataset1.time[0].values])
        time_coords_ds2 = np.array([dataset2.time[0].values])

        # Defining latitude and longitude coordinates for 1km resolution
        lat_res = 0.009  # ~1 km
        lon_res = 0.011  # ~1 km
        lat_coords = np.arange(dataset1['lat'].values.min(), dataset1['lat'].values.max(), lat_res)
        lon_coords = np.arange(dataset1['lon'].values.min(), dataset1['lon'].values.max(), lon_res)
        variables_to_interpolate = ["SWE_tavg"]

        # Parallelize dataset interpolation
        logger.debug("Starting parallel interpolation of datasets")
        results = Parallel(
            n_jobs=-1 if self.parallel_compute else 1, backend="threading"
        )(
            delayed(self.interpolate_dataset)(
                ds, variables_to_interpolate, lat_coords, lon_coords, time_coords
            )
            for ds, time_coords in [
                (dataset1, time_coords_ds1),
                (dataset2, time_coords_ds2),
            ]
        )

        new_ds1, new_ds2 = results
        logger.debug("Parallel interpolation for resolution completed")

        # Combine datasets and clip to Missouri Basin
        combined_dataset = xr.concat([new_ds1, new_ds2], dim="time")
        mo_basin = mo_basin.to_crs("EPSG:4326")
        combined_dataset = combined_dataset.rio.write_crs("EPSG:4326")
        logger.debug("Clipping combined dataset to Missouri Basin")
        
        # clipped_dataset = combined_dataset.rio.clip(
        #     mo_basin.geometry, all_touched=True, drop=True
        # )

        # Remove grid_mapping attributes
        for var in combined_dataset.data_vars:
            if "grid_mapping" in combined_dataset[var].attrs:
                del combined_dataset[var].attrs["grid_mapping"]

        # Save to NetCDF
        logger.debug(f"Saving combined dataset for resolution to {output_file}")
        combined_dataset.to_netcdf(output_file)
        end_time = time.time()
        logger.info(
            f"Combining the two resolution datasets successfully completed in {end_time - start_time:.2f} seconds."
        )

        return output_file, combined_dataset


    def calculate_eta(self, swe_change, threshold, k_value):
        """
        Calculate the efficiency value (eta) based on the SWE change.

        Args:
            swe_change (np.ndarray): The SWE change values
            threshold (float): The threshold value
            k_value (float): The k value

        Returns:
            np.ndarray: The efficiency values
        """
        # Apply logistic function directly, keeping zeros
        return 1 / (1 + np.exp(-k_value * (swe_change - threshold)))

    def generate_swe_difference(self, ds):
        """
        Generate the SWE difference dataset.

        Args:
            ds (xarray.Dataset): The dataset containing the SWE values

        Returns:
            output_file (str): The output file name
            swe_difference_dataset (xarray.Dataset): The new dataset
        """
        logger.info("Generating the SWE difference dataset.")
        swe = ds["SWE_tavg"]
        swe_masked = swe.where(~np.isnan(swe))
        swe_diff = swe_masked.diff(dim="time", label="lower")
        swe_diff_abs = abs(swe_diff)
        zero_diff = xr.full_like(swe.isel(time=0), fill_value=0)
        swe_diff_abs = xr.concat([zero_diff, swe_diff_abs], dim="time")
        swe_diff_abs = swe_diff_abs.assign_coords(time=swe["time"])
        T = 10
        k = 0.2
        eta5_values = self.calculate_eta(swe_diff_abs, T, k)
        eta5_values = eta5_values.where(~np.isnan(swe_masked), np.nan)
        eta5_values = eta5_values.broadcast_like(swe)
        eta5_da = xr.DataArray(
            eta5_values.values,
            coords={
                "time": swe["time"],
                "y": swe["y"],
                "x": swe["x"],
            },
            dims=["time", "y", "x"],
            name="eta5",
        )
        swe_difference_dataset = xr.Dataset(
            {"eta5": eta5_da, "swe_diff_abs": swe_diff_abs}
        ).transpose("time", "y", "x")
        last_date = str(swe["time"][-1].values)[:10].replace("-", "")
        output_file = os.path.join(
            self.current_simulation_date, f"Efficiency_SWE_Change_{last_date}.nc"
        )
        swe_difference_dataset.to_netcdf(output_file)
        logger.info("Generating the SWE difference dataset successfully completed.")
        return output_file, swe_difference_dataset

    def generate_surface_temp(self, ds):
        """
        Generate the surface temperature efficiency dataset.

        Args:
            ds (xarray.Dataset): The dataset containing the SWE and surface temperature values

        Returns:
            output_file (str): The output file name
            surface_temp_dataset (xarray.Dataset): The new dataset
        """
        logger.info("Generating surface temperature efficiency dataset.")
        swe = ds["SWE_tavg"]
        surface_temp = ds["AvgSurfT_tavg"] - 273.15
        temp_masked = surface_temp.where(~np.isnan(surface_temp))
        T = 0
        k = 0.5

        def calculate_eta0_temp(temp_values, threshold=T, k_value=k):
            # Apply logistic function
            # return 1 / (1 + np.exp(-k_value * (temp_values - threshold)))
            return 1 / (1 + np.exp(k_value * (temp_values - threshold)))

        eta0_values = calculate_eta0_temp(temp_masked)
        eta0_values = eta0_values.where(~np.isnan(surface_temp), np.nan)
        eta0_values = eta0_values.broadcast_like(surface_temp)
        eta0_da = xr.DataArray(
            eta0_values.values,
            coords={
                "time": surface_temp["time"],
                "y": surface_temp["y"],
                "x": surface_temp["x"],
            },
            dims=["time", "y", "x"],
            name="eta0",
        )
        surface_temp_dataset = xr.Dataset({"eta0": eta0_da})
        surface_temp_dataset = surface_temp_dataset.transpose("time", "y", "x")
        for var in surface_temp_dataset.variables:
            if "grid_mapping" in surface_temp_dataset[var].attrs:
                del surface_temp_dataset[var].attrs["grid_mapping"]
        last_date = str(swe["time"][-1].values)[:10].replace("-", "")
        output_file = os.path.join(
            self.current_simulation_date, f"Efficiency_SurfTemp_{last_date}.nc"
        )
        surface_temp_dataset.to_netcdf(output_file)
        logger.info("Generating surface temperature dataset successfully completed.")

        return output_file, surface_temp_dataset

    def generate_sensor_gcom(self, ds):
        """
        Generate the GCOM efficiency dataset.

        Args:
            ds (xarray.Dataset): The dataset containing the SWE values

        Returns:
            output_file (str): The output file name
            sensor_gcom_dataset (xarray.Dataset): The new dataset
        """
        logger.info("Generating GCOM efficiency dataset.")
        swe = ds["SWE_tavg"]
        T = 150
        k = 0.03
        epsilon = 0.05

        def calculate_eta2(swe_value, threshold=T, k_value=k, intercept=epsilon):
            # Logistic function with intercept
            return intercept + (1 - intercept) / (
                1 + np.exp(k_value * (swe_value - threshold))
            )

        swe_masked = swe.where(~np.isnan(swe))
        eta2_values = calculate_eta2(swe_masked)
        eta2_values = eta2_values.broadcast_like(swe)
        eta2_da = xr.DataArray(
            eta2_values.values,
            coords={
                "time": swe["time"],
                "y": swe["y"],
                "x": swe["x"],
            },
            dims=["time", "y", "x"],
            name="eta2",
        )
        sensor_gcom_dataset = xr.Dataset({"eta2": eta2_da}).transpose("time", "y", "x")
        for var in sensor_gcom_dataset.variables:
            if "grid_mapping" in sensor_gcom_dataset[var].attrs:
                del sensor_gcom_dataset[var].attrs["grid_mapping"]
        last_date = str(swe["time"][-1].values)[:10].replace("-", "")
        output_file = os.path.join(
            self.current_simulation_date, f"Efficiency_Sensor_GCOM_{last_date}.nc"
        )
        sensor_gcom_dataset.to_netcdf(output_file)
        logger.info("Generating GCOM efficiency dataset successfully completed.")

        return output_file, sensor_gcom_dataset

    def generate_sensor_capella(self, ds):
        """
        Generate the Capella efficiency dataset.

        Args:
            ds (xarray.Dataset): The dataset containing the SWE values

        Returns:
            output_file (str): The output file name
            sensor_capella_dataset (xarray.Dataset): The new dataset
        """
        logger.info("Generating Capella efficiency dataset.")
        swe = ds["SWE_tavg"]
        T = 150
        k = -0.03
        epsilon = 0.05

        def calculate_eta2(swe_value, threshold=T, k_value=k, intercept=epsilon):
            # Logistic function with intercept
            return intercept + (1 - intercept) / (
                1 + np.exp(k_value * (swe_value - threshold))
            )

        swe_masked = swe.where(~np.isnan(swe))
        eta2_values = calculate_eta2(swe_masked)
        eta2_values = eta2_values.broadcast_like(swe)
        eta2_da = xr.DataArray(
            eta2_values.values,
            coords={
                "time": swe["time"],
                "y": swe["y"],
                "x": swe["x"],
            },
            dims=["time", "y", "x"],
            name="eta2",
        )
        sensor_capella_dataset = xr.Dataset({"eta2": eta2_da}).transpose(
            "time", "y", "x"
        )
        for var in sensor_capella_dataset.variables:
            if "grid_mapping" in sensor_capella_dataset[var].attrs:
                del sensor_capella_dataset[var].attrs["grid_mapping"]
        last_date = str(swe["time"][-1].values)[:10].replace("-", "")
        
        output_file = os.path.join(
            self.current_simulation_date, f"Efficiency_Sensor_Capella_{last_date}.nc"
        )
        sensor_capella_dataset.to_netcdf(output_file)
        logger.info("Generating Capella efficiency dataset successfully completed.")

        return output_file, sensor_capella_dataset
    
    # MODIFIED BY DIVYA

    def generate_snowcover(self, ds):   
        """
        Generate the snow cover dataset.

        Args:
            ds (xarray.Dataset): The dataset containing the snow cover values

        Returns:
            output_file (str): The output file name
            snowcover_dataset (xarray.Dataset): The new dataset
        """
        logger.info("Generating snow cover dataset.")
        # check datavariables in the dataset    
        logger.info(f"Dataset variables: {ds.data_vars}")
        sc = ds["Snowcover_tavg"]
        snowcover_masked = sc.where(~np.isnan(sc))
        logger.info("Generating Snow_cover efficiency dataset.")
        T = 0.3
        k = 0.5

        
        last_date = str(sc["time"][-1].values)[:10].replace("-", "")
        snow_cover_file = os.path.join(
            self.current_simulation_date, f"Efficiency_snowcover_{last_date}.nc"
        )
        logger.info(f"Saving snow cover file to: {snow_cover_file}")

        eta_sc_values = self.calculate_eta(snowcover_masked, T, k)
        eta_sc_values.to_netcdf(snow_cover_file)

        return eta_sc_values, snow_cover_file

    # MODIFIED BY DIVYA
    def generate_resolution(self, ds, target_resolution_file, mo_basin):
        """
        Generate the resolution dataset.
        Args:
            ds (xarray.Dataset): The dataset containing the SWE values

        Returns:
            output_file (str): The output file name
            resolution_dataset (xarray.Dataset): The new dataset
        """
        logger.info("Generating resolution dataset.")
        ds = ds.rio.write_crs("EPSG:4326", inplace=False)
        res_x, res_y = ds.rio.resolution()
        res_x = abs(res_x)
        res_y = abs(res_y)
        factor = 50 # in km

        # Convert 50 km to degrees
        scale_x = res_x * factor
        scale_y = res_y * factor

        # New dimensions
        height = int((ds.rio.height * res_y) / scale_y)
        width = int((ds.rio.width * res_x) / scale_x)

        ds_50km = ds.rio.reproject(
        ds.rio.crs,
        shape=(int(height), int(width)),
        resampling=Resampling.bilinear
        )

        ds_1km = ds_50km.rio.reproject_match(ds)
        ds_abs = abs(ds_1km-ds)        
        k = 0.7
        T = np.nanmedian(ds_abs['SWE_tavg'].values)

        swe = ds["SWE_tavg"]
        last_date = str(swe["time"][-1].values)[:10].replace("-", "")
        resolution_file_nontaskable = os.path.join(
            self.current_simulation_date, f"efficiency_resolution_nontaskable{last_date}.nc"
        )

        eta_res_values = expit(-k * (ds_abs - T))
        

        resolution_file_taskable = os.path.join(
            self.current_simulation_date, "resolution_taskable.nc"
        )      

        # if os.path.exists(resolution_file_taskable):
        #     logger.info(
        #         f"File {resolution_file_taskable} already exists. Reading the existing file."
        #     )
        #     eta_res_values_taskable = xr.open_dataset(resolution_file_taskable)     
        # else:  
          
        eta_res_values_taskable = xr.ones_like(eta_res_values).astype("float32")        
    
        # Reprojecting to the standard resolution of the other layers
        target_resolution_file = target_resolution_file.rio.write_crs("EPSG:4326", inplace=False)
        eta_res_values = eta_res_values.rio.write_crs(
            "EPSG:4326", inplace=False
        )
        eta_res_values_taskable = eta_res_values_taskable.rio.write_crs(
            "EPSG:4326", inplace=False
        )
        
        logger.info("Reprojecting resolution datasets to match target resolution.")
        logger.info(f"taskable data {eta_res_values_taskable}")

        resolution_nontaskable_50km = eta_res_values.rio.reproject_match(target_resolution_file,Resampling=Resampling.bilinear) 
        resolution_taskable_50km = eta_res_values_taskable.rio.reproject_match(target_resolution_file,Resampling=Resampling.bilinear)

        logger.info("Reprojecting resolution datasets successfully completed.")

        resolution_taskable_50km = resolution_taskable_50km.rio.write_crs("EPSG:4326", inplace=False)
        resolution_taskable_50km = resolution_taskable_50km.rio.clip(mo_basin.geometry,all_touched=True, drop=True) 
        
        resolution_nontaskable_50km = resolution_nontaskable_50km.rio.write_crs("EPSG:4326", inplace=False)
        resolution_nontaskable_50km = resolution_nontaskable_50km.rio.clip(mo_basin.geometry,all_touched=True, drop=True) 

        # # Remove grid_mapping attributes
        # for var in resolution_taskable_50km.data_vars:
        #     if "grid_mapping" in resolution_taskable_50km[var].attrs:
        #         del resolution_taskable_50km[var].attrs["grid_mapping"] 

        # Saving as netcdf 
        resolution_nontaskable_50km.to_netcdf(resolution_file_nontaskable)
        resolution_taskable_50km.to_netcdf(resolution_file_taskable)

        return resolution_nontaskable_50km, resolution_taskable_50km,resolution_file_nontaskable   

      
    # def generate_sensor_capella(self, ds):
    #     """
    #     Generate the Capella efficiency dataset.

    #     Args:
    #         ds (xarray.Dataset): The dataset containing the SWE values

    #     Returns:
    #         output_file (str): The output file name
    #         capella_dataset (xarray.Dataset): The new dataset
    #     """
    #     logger.info("Generating Capella efficiency dataset.")
    #     swe = ds["SWE_tavg"]
    #     eta2_capella_values = xr.DataArray(
    #         np.ones_like(swe.values),
    #         coords={
    #             "time": swe["time"],
    #             "y": swe["y"],
    #             "x": swe["x"],
    #         },
    #         dims=["time", "y", "x"],
    #         name="eta2",
    #     )
    #     eta2_capella_values = eta2_capella_values.where(~np.isnan(swe), np.nan)
    #     capella_dataset = xr.Dataset({"eta2": eta2_capella_values}).transpose(
    #         "time", "y", "x"
    #     )
    #     for var in capella_dataset.variables:
    #         if "grid_mapping" in capella_dataset[var].attrs:
    #             del capella_dataset[var].attrs["grid_mapping"]
    #     last_date = str(swe["time"][-1].values)[:10].replace("-", "")
    #     output_file = os.path.join(
    #         self.current_simulation_date, f"Efficiency_Sensor_Capella_{last_date}.nc"
    #     )
    #     capella_dataset.to_netcdf(output_file)
    #     logger.info("Generating Capella efficiency dataset successfully completed.")
    #     return output_file, capella_dataset

    def combine_and_multiply_datasets(
        self, ds, eta5_file, eta0_file, eta2_file,eta_sc_file,eta_res_file, weights, output_file
    ):
        """
        Combine three datasets by applying weights and performing grid-cell multiplication.

        Args:
            ds (xarray.Dataset): The dataset containing the SWE values.
            eta5_file (str): Path to the NetCDF file for the eta5 dataset.
            eta0_file (str): Path to the NetCDF file for the eta0 dataset.
            eta2_file (str): Path to the NetCDF file for the eta2 dataset.
            weights (dict): Weights for eta5, eta0, and eta2 (e.g., {'eta5': 0.5, 'eta0': 0.3, 'eta2': 0.2}).
            output_file (str): Path to save the resulting combined dataset.

        Returns:
            output_file (str): The output file name
            combined_dataset (xarray.Dataset): The new dataset
        """
        logger.info("Combining and multiplying the datasets.")
        swe = ds["SWE_tavg"]
        eta5_ds = eta5_file
        eta0_ds = eta0_file
        eta2_ds = eta2_file
        eta_sc_ds = eta_sc_file
        eta_res_ds = eta_res_file
        eta5 = eta5_ds["eta5"]
        eta0 = eta0_ds["eta0"]
        eta2 = eta2_ds["eta2"]
        eta_sc = eta_sc_ds 
        eta_res= eta_res_ds["SWE_tavg"]
        weighted_eta5 = eta5 * weights["eta5"]
        weighted_eta0 = eta0 * weights["eta0"]
        weighted_eta2 = eta2 * weights["eta2"]
        weighted_eta_sc = eta_sc * weights["eta_sc"]
        weighted_eta_res = eta_res * weights["eta_res"]
        combined_values = weighted_eta5 * weighted_eta0 * weighted_eta2 * weighted_eta_sc * weighted_eta_res
        combined_dataset = xr.Dataset({"combined_eta": combined_values})
        combined_dataset["combined_eta"] = combined_dataset[
            "combined_eta"
        ].assign_coords({"time": eta5["time"], "y": eta5["y"], "x": eta5["x"]})
        last_date = str(swe["time"][-1].values)[:10].replace("-", "")
        output_file = os.path.join(
            self.current_simulation_date, f"{output_file}_{last_date}.nc"
        )
        combined_dataset.to_netcdf(output_file)
        logger.info("Combining and multiplying the datasets successfully completed.")

        return output_file, combined_dataset

    def process(self, gcom_ds, snowglobe_ds, mo_basin, start, end):
        """
        Process the datasets to generate the final efficiency dataset.

        Args:
            gcom_ds (xarray.Dataset): The GCOM dataset.
            snowglobe_ds (xarray.Dataset): The SnowGlobe dataset.
            mo_basin (GeoSeries): The Missouri Basin polygon.
            start (datetime): The start date.
            end (datetime): The end date.

        Returns:
            output_file (str): The output file name.
            final_eta_gdf (gpd.GeoDataFrame): The final efficiency GeoDataFrame.
        """
        logger.info("Generating final efficiency dataset.")
        duration = timedelta(days=1)
        frame_duration = timedelta(days=1)
        num_frames = int(1 + (end - start) / duration)
        roll_angle = (30 + 33.5) / 2
        roll_range = 33.5 - 30
        logger.info("Specifying constellation.")
        constellation = WalkerConstellation(
            name="SnowGlobe Ku",
            orbit=SunSynchronousOrbit(
                altitude=555e3,
                equator_crossing_time="06:00:30",
                equator_crossing_ascending=False,
                epoch=datetime(2019, 3, 1, tzinfo=timezone.utc),
            ),
            number_planes=1,
            number_satellites=5,
            instruments=[
                PointedInstrument(
                    name="SnowGlobe Ku-SAR",
                    roll_angle=-roll_angle,
                    field_of_regard=2 * roll_angle
                    + swath_width_to_field_of_regard(555e3, 50e3),
                    along_track_field_of_view=swath_width_to_field_of_view(
                        555e3, 50e3, 0
                    ),
                    cross_track_field_of_view=roll_range
                    + swath_width_to_field_of_view(555e3, 50e3, roll_angle),
                    is_rectangular=True,
                )
            ],
        )
        logger.info("Specifying constellation successfully completed.")
        time_step = timedelta(seconds=5)
        sim_times = pd.date_range(start, end + duration, freq=time_step)
        # logger.info("Computing orbit tracks.")
        # orbit_tracks = pd.concat(
        #     [
        #         collect_orbit_track(
        #             satellite=satellite,
        #             times=sim_times,
        #             mask=self.polygons[0],
        #         )
        #         for satellite in constellation.generate_members()
        #     ]
        # )
        # logger.info("Computing orbit tracks successfully completed.")
        logger.info("Computing ground tracks (P1).")
        start_time = time.time()
        ground_tracks = pd.concat(
            Parallel(n_jobs=-1 if self.parallel_compute else 1)(
                delayed(compute_ground_track)(
                    satellite=satellite,
                    times=sim_times,
                    # mask=self.polygons[0],
                    crs="spice",
                )
                for satellite in constellation.generate_members()
            ),
            ignore_index=True,
        )
        end_time = time.time()
        # logger.info("Computing ground tracks (P1) successfully completed.")
        logger.info(
            f"Computing ground tracks (P1) successfully completed in {end_time - start_time:.2f} seconds."
        )
        domain = box(-114, 37, -90, 50)
        amsr2 = PointedInstrument(
            name="AMSR2",
            cross_track_field_of_view=utils.swath_width_to_field_of_regard(
                700e3, 1450e3
            ),
            along_track_field_of_view=utils.swath_width_to_field_of_regard(
                700e3, 10e3 * 10
            ),  # 10x to allow longer time step
            req_target_sunlit=False,  # restrict to descending (nighttime) overpasses
            is_rectangular=True,
        )

        # orbit from https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle
        # note: tle MUST have an epoch close to start of mission period (use special data request if necessary)
        gcom_w = Satellite(
            name="GCOM-W",
            orbit=TwoLineElements(
                tle=[
                    "1 38337U 12025A   25001.47767252  .00002770  00000+0  62493-3 0  9994",
                    "2 38337  98.2252 304.6936 0001249  63.3270  72.0663 14.57087617671602",
                ]
            ),
            instruments=[amsr2],
        )
        logger.info("Computing ground tracks (P2).")
        start_time = time.time()
        gcom_tracks = pd.concat(
            Parallel(n_jobs=-1 if self.parallel_compute else 1)(
                delayed(compute_ground_track)(
                    gcom_w,
                    pd.date_range(
                        start + frame * frame_duration,
                        start + (frame + 1) * frame_duration,
                        freq=time_step,
                        inclusive="left",
                    ),
                    crs="spice",
                    mask=domain,
                )
                for frame in range(num_frames)
            ),
            ignore_index=True,
        ).sort_values(by="time")
        end_time = time.time()
        logger.info(
            f"Computing ground tracks (P2) successfully completed in {end_time - start_time:.2f} seconds."
        )
        gcom_tracks["time"] = pd.to_datetime(gcom_tracks["time"]).dt.tz_localize(None)
        gcom_eta = gcom_ds["combined_eta"].isel(time=1).rio.write_crs("EPSG:4326")
        snowglobe_eta = (
            snowglobe_ds["combined_eta"].isel(time=1).rio.write_crs("EPSG:4326")
        )
        gcom_union = gcom_tracks.union_all()
        snowglobe_union = ground_tracks.union_all()
        gcom_geometries = (
            [gcom_union]
            if gcom_union.geom_type == "Polygon"
            else list(gcom_union.geoms)
        )
        snowglobe_geometries = (
            [snowglobe_union]
            if snowglobe_union.geom_type == "Polygon"
            else list(snowglobe_union.geoms)
        )
        gcom_mask = xr.full_like(snowglobe_eta, False, dtype=bool)
        snowglobe_mask = xr.full_like(snowglobe_eta, False, dtype=bool)
        gcom_mask.values = geometry_mask(
            geometries=gcom_geometries,
            out_shape=snowglobe_eta.shape,
            transform=snowglobe_eta.rio.transform(),
            invert=True,
        )
        snowglobe_mask.values = geometry_mask(
            geometries=snowglobe_geometries,
            out_shape=snowglobe_eta.shape,
            transform=snowglobe_eta.rio.transform(),
            invert=True,
        )
        logger.info("Creating masks for ground tracks successfully completed.")
        logger.info("Computing final efficiency values.")
        final_eta = xr.full_like(snowglobe_eta, np.nan)
        final_eta = final_eta.where(snowglobe_mask)
        final_eta = final_eta.where(~gcom_mask, snowglobe_eta - gcom_eta)
        final_eta = final_eta.where(~snowglobe_mask, snowglobe_eta)
        final_eta = final_eta.where(snowglobe_mask, np.nan)
        final_last_date = snowglobe_eta["time"].values
        final_eta = final_eta.assign_coords(time=final_last_date)
        final_eta_df = final_eta.to_dataframe(name="final_eta").reset_index()
        final_eta_df = final_eta_df.dropna(subset=["x", "y", "final_eta"])
        x_res = abs(final_eta["x"].diff(dim="x").mean().values)
        y_res = abs(final_eta["y"].diff(dim="y").mean().values)

        polygons = []
        for row in final_eta_df.itertuples():
            x_center, y_center = row.x, row.y
            left = x_center - x_res / 2
            right = x_center + x_res / 2
            bottom = y_center - y_res / 2
            top = y_center + y_res / 2
            polygons.append(box(left, bottom, right, top))
        final_eta_gdf = gpd.GeoDataFrame(
            final_eta_df,
            geometry=polygons,
            crs="EPSG:4326",
        )
        final_eta_gdf["time"] = pd.Timestamp(final_last_date)
        
        output_file = os.path.join(
            self.current_simulation_date,
            f"Reward_{pd.Timestamp(final_last_date).strftime('%Y%m%d')}.geojson",
        )
        final_eta_gdf.to_file(output_file, driver="GeoJSON")
        logger.info("Generating final efficiency dataset successfully completed.")
        return output_file, final_eta_gdf

    def find_optimal_solution(self, final_eta_gdf):
        """
        Find the optimal solution.

        Args:
            final_eta_gdf (gpd.GeoDataFrame): The final efficiency GeoDataFrame.

        Returns:
            output_geojson (str): The output GeoJSON file name.
            selected_blocks_gdf (gpd.GeoDataFrame): The selected blocks GeoDataFrame.
        """
        unique_time = pd.Timestamp(final_eta_gdf["time"].iloc[0])
        N = 50
        final_eta_gdf["final_eta"] = final_eta_gdf["final_eta"].replace(
            [np.inf, -np.inf], np.nan
        )
        final_eta_gdf = final_eta_gdf.dropna(subset=["final_eta"])
        model = LpProblem("Final_Eta_Optimization", LpMaximize)
        x = {i: LpVariable(f"x_{i}", cat="Binary") for i in final_eta_gdf.index}
        objective = lpSum(
            x[i] * final_eta_gdf.loc[i, "final_eta"] for i in final_eta_gdf.index
        )
        model += objective
        model += lpSum(x[i] for i in final_eta_gdf.index) <= N, "Max_Selections"
        model.solve()
        if value(model.objective) is not None:
            logger.info("Optimal solution found.")
            selected_blocks = []

            for i in final_eta_gdf.index:
                if x[i].value() > 0.5:
                    selected_blocks.append(
                        {
                            "geometry": final_eta_gdf.loc[i, "geometry"],
                            "final_eta": final_eta_gdf.loc[i, "final_eta"],
                            "time": unique_time,  # Assign the time value
                        }
                    )
            selected_blocks_gdf = gpd.GeoDataFrame(selected_blocks, crs="EPSG:4326")
            output_geojson = os.path.join(
                self.current_simulation_date,
                f"Selected_Cells_Optimization_{unique_time.strftime('%Y%m%d')}.geojson",
            )
            selected_blocks_gdf.to_file(output_geojson, driver="GeoJSON")
            logger.info(f"Optimization output saved as {output_geojson}")

            logger.info(
                f"Selected cells saved to '{output_geojson}' with time: {unique_time}"
            )
        else:
            logger.info("No optimal solution found.")
        return output_geojson, selected_blocks_gdf

    def downsample_array(self, array, downsample_factor):
        """
        Downsamples the given array by the specified factor.

        Args:
            array (np.ndarray): The array to downsample.
            downsample_factor (int): The factor by which to downsample the array.

        Returns:
            np.ndarray: The downsampled array.
        """
        return array[::downsample_factor, ::downsample_factor]

    def get_extents(self, dataset, variable):
        """
        Get the extents of the dataset.

        Args:
            dataset (xarray.Dataset): The dataset
            variable (str): The variable name

        Returns:
            top_left (tuple): The top left corner coordinates
            top_right (tuple): The top right corner coordinates
            bottom_left (tuple): The bottom left corner coordinates
            bottom_right (tuple): The bottom right corner coordinates
        """
        geo_transform = dataset["spatial_ref"].GeoTransform.split()
        geo_transform = [float(value) for value in geo_transform]
        min_x = geo_transform[0]
        pixel_width = geo_transform[1]
        max_y = geo_transform[3]
        pixel_height = geo_transform[5]
        n_rows, n_cols = dataset[variable][0, :, :].shape
        top_left = (min_x, max_y)
        top_right = (min_x + n_cols * pixel_width, max_y)
        bottom_left = (min_x, max_y + n_rows * pixel_height)
        bottom_right = (min_x + n_cols * pixel_width, max_y + n_rows * pixel_height)
        return top_left, top_right, bottom_left, bottom_right

    def encode(
        self,
        dataset,
        variable,
        output_path,
        time_step,
        scale,
        geojson_path,
        downsample_factor=1,
        rotate=False,
    ):
        """
        Encode the raster layer as a PNG image.

        Args:
            dataset (xarray.Dataset): The dataset
            variable (str): The variable name
            output_path (str): The output path
            time_step (int): The time step
            scale (float): The scale factor
            geojson_path (str): The GeoJSON path
            downsample_factor (int): The downsample factor
            rotate (bool): If True, rotate the image

        Returns:
            raster_layer_encoded (str): The encoded raster layer
            top_left (tuple): The top left corner coordinates
            top_right (tuple): The top right corner coordinates
            bottom_left (tuple): The bottom left corner coordinates
            bottom_right (tuple): The bottom right corner coordinates
        """
        raster_layer = dataset[variable]

        raster_layer = raster_layer.rio.write_crs("EPSG:4326")
        clipped_layer = raster_layer.rio.clip(self.polygons, all_touched=True)
        raster_layer = clipped_layer.isel(time=time_step)
        raster_layer_min = np.nanmin(raster_layer)
        raster_layer_max = np.nanmax(raster_layer)

        na_mask = np.isnan(raster_layer)

        if raster_layer_max > raster_layer_min:
            normalized_layer = (raster_layer - raster_layer_min) / (
                raster_layer_max - raster_layer_min
            )
        else:
            normalized_layer = np.zeros_like(raster_layer)

        colormap = plt.get_cmap("Blues_r")
        rgba_image = colormap(normalized_layer)

        rgba_image[..., 3] = np.where(na_mask, 0, 1)

        rgba_image = (rgba_image * 255).astype(np.uint8)

        if rotate:
            # Rotate the image about the x-axis by 180 degrees
            rgba_image = np.flipud(rgba_image)

        image = Image.fromarray(rgba_image, "RGBA")
        image.save(os.path.join(self.current_simulation_date, output_path))
        try:
            if rotate:
                bottom_left, bottom_right, top_left, top_right = self.get_extents(
                    dataset, variable=variable
                )
            else:
                top_left, top_right, bottom_left, bottom_right = self.get_extents(
                    dataset, variable=variable
                )
        except:
            top_left = top_right = bottom_left = bottom_right = None

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")

        raster_layer_encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # logger.info('Encoding snow layer successfully completed.')

        return raster_layer_encoded, top_left, top_right, bottom_left, bottom_right


    def find_most_recent_file(self, bucket_name, directories, file_name_pattern):
        """
        Search for the most recent matching file across provided S3 directories,
        prioritizing assimilation subdirectories starting with 'out_'.

        Args:
            s3: Boto3 S3 client
            bucket_name (str): Name of the S3 bucket
            directories (list): List of top-level S3 prefixes to search
            file_name_pattern (str): Expected file name suffix (e.g., 'LIS_HIST_*.nc')

        Returns:
            str or None: Key of the most recent matching file if found, else None
        """
        paginator = self.s3.get_paginator("list_objects_v2")

        # Priority 1: assimilation (out_ subdirs)
        for directory in directories:
            logger.debug(f"Searching in directory: {directory}")
            if "assimilation" in directory:
                logger.info(
                    f"Looking for subdirectories in assimilation path: {directory}"
                )
                pages = paginator.paginate(
                    Bucket=bucket_name, Prefix=directory, Delimiter="/"
                )

                # subdirs = [
                #     prefix["Prefix"]
                #     for page in pages
                #     for prefix in page.get("CommonPrefixes", [])
                #     if os.path.basename(prefix["Prefix"].rstrip("/")).startswith("out_")
                # ]

                # Not considering out_ prefix for subdirectories
                subdirs = [
                    prefix["Prefix"]
                    for page in pages
                    for prefix in page.get("CommonPrefixes", [])
                ]

                logger.info(f"Assimilation subdirs found: {subdirs}")

                if subdirs:
                    most_recent_subdir = max(subdirs)
                    logger.info(
                        f"Most recent assimilation subdir: {most_recent_subdir}"
                    )

                    pages = paginator.paginate(
                        Bucket=bucket_name, Prefix=most_recent_subdir
                    )
                    for page in pages:
                        for obj in page.get("Contents", []):
                            logger.debug(f"Found object in assimilation: {obj['Key']}")
                            if obj["Key"].endswith(file_name_pattern):
                                logger.info(
                                    f"Found matching file in assimilation: {obj['Key']}"
                                )
                                return obj["Key"]

        logger.warning("No matching file found in assimilation.")

        # Priority 2: open_loop
        for directory in directories:
            if "open_loop" in directory:
                logger.info(f"Looking into open_loop directory: {directory}")
                pages = paginator.paginate(Bucket=bucket_name, Prefix=directory)
                for page in pages:
                    for obj in page.get("Contents", []):
                        logger.debug(f"Found object in open_loop: {obj['Key']}")
                        if obj["Key"].endswith(file_name_pattern):
                            logger.info(
                                f"Found matching file in open_loop: {obj['Key']}"
                            )
                            return obj["Key"]

        logger.warning(f"No matching file found for pattern: {file_name_pattern}")
        return None

    def download_file(
        self,
        bucket_name,
        file_name_pattern,
        local_filename=None,
        check_interval_sec=None,
        max_attempts=None,
    ):
        """
        Download a file by first checking assimilation (up to max_attempts), then falling back to open_loop.

        Args:
            bucket_name (str): The S3 bucket name.
            file_name_pattern (str): File name pattern.
            local_filename (str): Local name to save the file as.
            check_interval_sec (int): Time between retries.
            max_attempts (int): Max attempts for assimilation.

        Returns:
            xarray.Dataset: The loaded dataset.
        """
        # Get values from environment variables if not provided
        if check_interval_sec is None:
            check_interval_sec = int(os.environ.get("DOWNLOAD_CHECK_INTERVAL", 10))

        if max_attempts is None:
            max_attempts = int(os.environ.get("DOWNLOAD_MAX_ATTEMPTS", 0))

        logger.info(
            f"Using check_interval_sec={check_interval_sec}, max_attempts={max_attempts} for downloads"
        )

        # Try assimilation first (wait up to max_attempts)
        assimilation_dirs = ["inputs/LIS/assimilation/"]
        open_loop_dirs = ["inputs/LIS/open_loop_PMWonly/"]

        file_key = None
        for attempt in range(max_attempts):
            logger.info(
                f"[Assimilation] Attempt {attempt + 1} to find file {file_name_pattern}"
            )
            file_key = self.find_most_recent_file(
                bucket_name, assimilation_dirs, file_name_pattern
            )
            if file_key:
                break
            if attempt < max_attempts - 1:
                logger.warning(
                    f"File not found in assimilation. Retrying in {check_interval_sec / 60:.0f} minutes..."
                )
                time.sleep(check_interval_sec)

        # Fallback to open loop if not found
        if not file_key:
            logger.warning("Assimilation file not found. Trying open_loop.")
            file_key = self.find_most_recent_file(
                bucket_name, open_loop_dirs, file_name_pattern
            )

        if not file_key:
            raise FileNotFoundError(
                f"File {file_name_pattern} not found in assimilation or open_loop after {max_attempts} attempts."
            )

        if local_filename is None:
            local_filename = os.path.basename(file_key)

        if not os.path.exists(local_filename):
            logger.info(f"Downloading {file_key} to {local_filename}...")
            config = TransferConfig(
                use_threads=True if self.parallel_compute else False
            )
            self.s3.download_file(
                Bucket=bucket_name, Key=file_key, Filename=local_filename, Config=config
            )
        else:
            logger.info(f"File already exists locally: {local_filename}")

        return xr.open_dataset(local_filename, engine="h5netcdf")

    def download_geojson(self, key, filename, bucket="snow-observing-systems"):
        """
        Download a GeoJSON file from an S3 bucket

        Args:
            s3: S3 client
            bucket: S3 bucket name
            key: S3 object key
            filename: Filename to save the file as

        Returns:
            dataset (gpd.GeoDataFrame): The GeoDataFrame
        """
        if not os.path.isfile(filename):
            logger.info(f"Downloading file from S3: {filename}")
            config = TransferConfig(
                use_threads=True if self.parallel_compute else False
            )
            self.s3.download_file(
                Bucket=bucket, Key=key, Filename=filename, Config=config
            )
        else:
            logger.info(f"File already exists: {filename}")

        dataset = gpd.read_file(filename)
        return dataset

    def upload_file(self, key, filename, bucket="snow-observing-systems"):
        """
        Upload a file to an S3 bucket

        Args:
            s3: S3 client
            bucket: S3 bucket name
            key: S3 object key
            filename: Filename to upload
        """
        logger.info(f"Uploading file to S3.")
        config = TransferConfig(use_threads=True if self.parallel_compute else False)
        self.s3.upload_file(Filename=filename, Bucket=bucket, Key=key, Config=config)
        logger.info(f"Uploading file to S3 successfully completed.")

    def detect_level_change(self, new_value, old_value, level):
        """
        Detect a change in the level of the time value (day, week, or month).

        Args:
            new_value (datetime): New time value
            old_value (datetime): Old time value
            level (str): Level of time value to detect changes ('day', 'week', or 'month')

        Returns:
            bool: True if the level has changed, False otherwise
        """
        if level == "day":
            return new_value.date() != old_value.date()
        elif level == "week":
            return new_value.isocalendar()[1] != old_value.isocalendar()[1]
        elif level == "month":
            return new_value.month != old_value.month
        else:
            raise ValueError("Invalid level. Choose from 'day', 'week', or 'month'.")

    def multipolygon_to_polygon(self, geometry):
        """
        Convert a MultiPolygon to a Polygon

        Args:
            geometry (shapely.geometry.MultiPolygon): The MultiPolygon

        Returns:
            shapely.geometry.Polygon: The Polygon
        """
        if geometry.geom_type == "MultiPolygon":
            # Combine all polygons into a single polygon
            return Polygon(
                [
                    coord
                    for polygon in geometry.geoms
                    for coord in polygon.exterior.coords
                ]
            )
        return geometry

    def process_geojson(self, mo_basin):
        """
        Process the Missouri Basin GeoDataFrame

        Args:
            mo_basin (gpd.GeoDataFrame): The Missouri Basin GeoDataFrame

        Returns:
            gpd.GeoSeries: The GeoSeries
        """
        mo_basin.at[0, "geometry"] = self.multipolygon_to_polygon(
            mo_basin.at[0, "geometry"]
        )
        return gpd.GeoSeries(
            Polygon(mo_basin.iloc[0].geometry.exterior), crs="EPSG:4326"
        )

    def on_change(self, source, property_name, old_value, new_value):
        """
        Handle changes to properties

        Args:
            source (object): The source object
            property_name (str): The property name
            old_value (object): The old value
            new_value (object): The new value
        """
        if property_name == "time":

            # Determine if day has changed
            change = self.detect_level_change(new_value, old_value, "day")

            # Publish message if day, week, or month has changed
            if change:

                self.current_simulation_date = os.path.join(
                    self.output_directory, str(new_value.date())
                )
                self.data_utils.create_directories([self.current_simulation_date])
                old_value_reformat = str(old_value.date()).replace("-", "")
                new_value_reformat = str(new_value.date()).replace("-", "")

                # Establish connection to S3
                s3 = AWSUtils().client
                self.s3 = s3

                mo_basin = self.download_geojson(
                    # s3=s3,
                    # bucket="snow-observing-systems",
                    key="inputs/vector/WBDHU2_4326.geojson",
                    # filename="WBDHU2_4326.geojson",
                    filename="inputs/vector/WBDHU2_4326.geojson",
                )
                mo_basin = self.process_geojson(mo_basin)
                self.polygons = mo_basin.geometry

                # Combined dataset#
                # Get first dataset
                # dataset1 = self.download_file(
                #     s3=s3,
                #     bucket="snow-observing-systems",
                #     key=f"LIS_HIST_{old_value_reformat}0000.d01.nc", #f"inputs/LIS/open_loop/LIS_HIST_{old_value_reformat}0000.d01.nc"
                #     filename=os.path.join(
                #         self.input_directory,
                #         f"LIS_HIST_{old_value_reformat}0000.d01.nc",
                #     ),
                # )
                # # Get second dataset
                # dataset2 = self.download_file(
                #     s3=s3,
                #     bucket="snow-observing-systems",
                #     key=f"LIS_HIST_{new_value_reformat}0000.d01.nc", #f"inputs/LIS/open_loop/LIS_HIST_{new_value_reformat}0000.d01.nc"
                #     filename=os.path.join(
                #         self.input_directory,
                #         f"LIS_HIST_{new_value_reformat}0000.d01.nc",
                #     ),
                # )
                
                # New Combined dataset
                dataset1 = self.download_file(
                    bucket_name="snow-observing-systems",
                    file_name_pattern=f"LIS_HIST_{old_value_reformat}0000.d01.nc",
                    local_filename=os.path.join(
                        self.input_directory,
                        f"LIS_HIST_{old_value_reformat}0000.d01.nc",
                    ),
                )

                dataset2 = self.download_file(
                    bucket_name="snow-observing-systems",
                    file_name_pattern=f"LIS_HIST_{new_value_reformat}0000.d01.nc",
                    local_filename=os.path.join(
                        self.input_directory,
                        f"LIS_HIST_{new_value_reformat}0000.d01.nc",
                    ),
                )
                # Generate the combined dataset
                combined_output_file, combined_dataset = self.generate_combined_dataset(
                    dataset1, dataset2, mo_basin
                )
                # Upload dataset to S3
                self.upload_file(
                    key=combined_output_file, filename=combined_output_file
                )
                # MODIFIED BY DIVYA 
                # Resolution layer processing
                combined_output_file_resolution, combined_dataset_resolution = self.generate_combined_dataset_resolution(
                    dataset1, dataset2, mo_basin
                )
                # Upload dataset to S3
                self.upload_file(
                    key=combined_output_file_resolution, filename=combined_output_file_resolution
                )


                # Select the SWE_tavg variable for a specific time step (e.g., first time step)
                swe_data = combined_dataset["SWE_tavg"].isel(time=1)  # SEND AS MESSAGE
                swe_layer_encoded, top_left, top_right, bottom_left, bottom_right = (
                    self.encode(
                        dataset=combined_dataset,
                        variable="SWE_tavg",
                        output_path=f"swe_data_{new_value_reformat}.png",
                        time_step=1,  # 0,
                        scale="time",
                        geojson_path="WBD_10_HU2_4326.geojson",
                        rotate=True,
                    )
                )
                if self.visualize_all_layers:
                    self.app.send_message(
                        self.app.app_name,
                        "layer",
                        SWEChangeLayer(
                            swe_change_layer=swe_layer_encoded,
                            top_left=top_left,
                            top_right=top_right,
                            bottom_left=bottom_left,
                            bottom_right=bottom_right,
                        ).model_dump_json(),
                    )
                    logger.info("Publishing message successfully completed.")
                    time.sleep(15)
               

            
                # ETA5 dataset#
                # Generate the SWE difference
                swe_output_file, eta5_file = self.generate_swe_difference(
                    ds=combined_dataset
                )
                # Upload dataset to S3
                self.upload_file(key=swe_output_file, filename=swe_output_file)
                # Select the eta5 variable for a specific time step (e.g., first time step)
                eta5_data = eta5_file["eta5"].isel(time=1)
                eta5_layer_encoded, _, _, _, _ = self.encode(
                    dataset=eta5_file,
                    variable="eta5",
                    output_path=f"eta5_data_{new_value_reformat}.png",
                    time_step=1,
                    scale="time",
                    geojson_path="WBD_10_HU2_4326.geojson",
                    rotate=True,
                )
                if self.visualize_swe_change:
                    self.app.send_message(
                        self.app.app_name,
                        "layer",
                        SWEChangeLayer(
                            swe_change_layer=eta5_layer_encoded,
                            top_left=top_left,
                            top_right=top_right,
                            bottom_left=bottom_left,
                            bottom_right=bottom_right,
                        ).model_dump_json(),
                    )
                    logger.info("Publishing message successfully completed.")

               
                # ETA0 dataset
                # Generate the surface temperature dataset
                surfacetemp_output_file, eta0_file = self.generate_surface_temp(
                    ds=combined_dataset
                )
                # Upload dataset to S3
                self.upload_file(
                    key=surfacetemp_output_file, filename=surfacetemp_output_file
                )

                # MODIFIED BY DIVYA - Resolution - adding here as I need eta0_file for resolution
                # Generate the resolution dataset
                resolution_dataset_nontaskable_eta, resolution_dataset_taskable_eta, resolution_output_file = (
                    self.generate_resolution(
                        ds=combined_dataset_resolution,target_resolution_file = eta5_file, mo_basin=mo_basin
                    )
                )

                # Upload dataset to S3
                self.upload_file(
                    key=resolution_output_file, filename=resolution_output_file
                )

                # Generate the snow cover dataset
                eta_sc_values, eta_sc_file = (
                    self.generate_snowcover(
                        ds=combined_dataset
                    )
                )

                # Upload dataset to S3
                self.upload_file(
                    key=eta_sc_file, filename=eta_sc_file
                )

                # COMMENT BY DIVYA - will add layer- encoding if required later    
                # Use snow_cover_eta_file , resolution_dataset_nontaskable_eta, resolution_dataset_taskable_eta for further steps      
                


                # Select the eta0 variable for a specific time step (e.g., first time step)
                eta0_data = eta0_file["eta0"].isel(time=1)
                eta0_layer_encoded, _, _, _, _ = self.encode(
                    dataset=eta0_file,
                    variable="eta0",
                    output_path=f"eta0_data_{new_value_reformat}.png",
                    time_step=1,
                    scale="time",
                    geojson_path="WBD_10_HU2_4326.geojson",
                    rotate=True,
                )
                if self.visualize_all_layers:
                    self.app.send_message(
                        self.app.app_name,
                        "layer",
                        SWEChangeLayer(
                            swe_change_layer=eta0_layer_encoded,
                            top_left=top_left,
                            top_right=top_right,
                            bottom_left=bottom_left,
                            bottom_right=bottom_right,
                        ).model_dump_json(),
                    )
                    logger.info("Publishing message successfully completed.")
                    time.sleep(15)

               
                # ETA2 GCOM dataset
                # Generate the sensor GCOM dataset
                sensor_gcom_output_file, eta2_file_GCOM = self.generate_sensor_gcom(
                    ds=combined_dataset
                )
                # Upload dataset to S3
                self.upload_file(
                    key=sensor_gcom_output_file, filename=sensor_gcom_output_file
                )
                # Select the eta2 variable for a specific time step (e.g., first time step)
                eta2_data_GCOM = eta2_file_GCOM["eta2"].isel(time=1)
                eta2_gcom_layer_encoded, _, _, _, _ = self.encode(
                    dataset=eta2_file_GCOM,
                    variable="eta2",
                    output_path=f"eta2_gcom_data_{new_value_reformat}.png",
                    time_step=1,
                    scale="time",
                    geojson_path="WBD_10_HU2_4326.geojson",
                    rotate=True,
                )
                if self.visualize_all_layers:
                    self.app.send_message(
                        self.app.app_name,
                        "layer",
                        SWEChangeLayer(
                            swe_change_layer=eta2_gcom_layer_encoded,
                            top_left=top_left,
                            top_right=top_right,
                            bottom_left=bottom_left,
                            bottom_right=bottom_right,
                        ).model_dump_json(),
                    )
                    logger.info("Publishing message successfully completed.")
                    time.sleep(15)


                # ETA2 Capella dataset
                # Generate the sensor capella dataset
                sensor_capella_output_file, eta2_file_Capella = (
                    self.generate_sensor_capella(ds=combined_dataset)
                )
                # Upload dataset to S3
                self.upload_file(
                    key=sensor_capella_output_file, filename=sensor_capella_output_file
                )
                # Select the eta2 variable for a specific time step (e.g., first time step)
                eta2_data_Capella = eta2_file_Capella["eta2"].isel(time=1)
                eta2_capella_layer_encoded, _, _, _, _ = self.encode(
                    dataset=eta2_file_Capella,
                    variable="eta2",
                    output_path=f"eta2_capella_data_{new_value_reformat}.png",
                    time_step=1,
                    scale="time",
                    geojson_path="WBD_10_HU2_4326.geojson",
                    rotate=True,
                )
                if self.visualize_all_layers:
                    self.app.send_message(
                        self.app.app_name,
                        "layer",
                        SWEChangeLayer(
                            swe_change_layer=eta2_capella_layer_encoded,
                            top_left=top_left,
                            top_right=top_right,
                            bottom_left=bottom_left,
                            bottom_right=bottom_right,
                        ).model_dump_json(),
                    )
                    logger.info("Publishing message successfully completed.")
                    time.sleep(15)

                # GCOM Final ETA
                # Define the weights for each dataset
                #weights = {"eta5": 0.5, "eta0": 0.3, "eta2": 0.2}
                weights = {"eta5": 0.5, "eta0": 0.3, "eta2": 0.2, "eta_sc": 0.3, "eta_res": 0.2}
                # Process GCOM datasets
                gcom_combine_multiply_output_file, gcom_dataset = (
                    self.combine_and_multiply_datasets(
                        ds=combined_dataset,
                        eta5_file=eta5_file,
                        eta0_file=eta0_file,
                        eta2_file=eta2_file_GCOM,
                        eta_sc_file = eta_sc_values,
                        eta_res_file = resolution_dataset_nontaskable_eta,
                        weights=weights,
                        output_file="Combined_Efficiency_Weighted_Product_GCOM",
                    )
                )

                # MODIFIED BY DIVYA - Combine and multiply files - commenting out for now

                # # Define the weights for each dataset
                # weights = {"eta5": 0.4, "eta0": 0.2, "eta2": 0.2, "snow_cover_eta": 0.1, "resolution_eta": 0.1}
                # # Process GCOM datasets
                # gcom_combine_multiply_output_file, gcom_dataset = (
                #     self.combine_and_multiply_datasets(
                #         ds=combined_dataset,
                #         eta5_file=eta5_file,
                #         eta0_file=eta0_file,
                #         eta2_file=eta2_file_GCOM,
                #         resolution_eta_file=resolution_dataset_nontaskable,
                #         snow_cover_eta_file=snow_cover_eta_file,
                #         weights=weights,
                #         output_file="Combined_Efficiency_Weighted_Product_GCOM",
                #     )
                # )


                # Upload dataset to S3
                self.upload_file(
                    key=gcom_combine_multiply_output_file,
                    filename=gcom_combine_multiply_output_file,
                )
                # Select the combined_eta variable for a specific time step (e.g., first time step)
                gcom_eta = gcom_dataset["combined_eta"].isel(time=1)

                gcom_eta_layer_encoded, _, _, _, _ = self.encode(
                    dataset=gcom_dataset,
                    variable="combined_eta",
                    output_path=f"gcom_eta_combined_data_{new_value_reformat}.png",
                    time_step=1,
                    scale="time",
                    geojson_path="WBD_10_HU2_4326.geojson",
                    rotate=True,
                )
                if self.visualize_all_layers:
                    self.app.send_message(
                        self.app.app_name,
                        "layer",
                        SWEChangeLayer(
                            swe_change_layer=gcom_eta_layer_encoded,
                            top_left=top_left,
                            top_right=top_right,
                            bottom_left=bottom_left,
                            bottom_right=bottom_right,
                        ).model_dump_json(),
                    )
                    logger.info("Publishing message successfully completed.")
                    time.sleep(15)

                # Capella Final ETA
                # Process Capella datasets
                capella_combine_multiply_output_file, capella_dataset = (
                    self.combine_and_multiply_datasets(
                        ds=combined_dataset,
                        eta5_file=eta5_file,
                        eta0_file=eta0_file,
                        eta2_file=eta2_file_Capella,
                        eta_sc_file = eta_sc_values,
                        eta_res_file = resolution_dataset_taskable_eta,
                        weights=weights,
                        output_file="Combined_Efficiency_Weighted_Product_Capella",
                    )
                )
                # Upload dataset to S3
                self.upload_file(
                    key=capella_combine_multiply_output_file,
                    filename=capella_combine_multiply_output_file,
                )
                capella_eta_layer_encoded, _, _, _, _ = self.encode(
                    dataset=capella_dataset,
                    variable="combined_eta",
                    output_path=f"capella_eta_combined_data_{new_value_reformat}.png",
                    time_step=1,
                    scale="time",
                    geojson_path="WBD_10_HU2_4326.geojson",
                    rotate=True,
                )
                if self.visualize_all_layers:
                    self.app.send_message(
                        self.app.app_name,
                        "layer",
                        SWEChangeLayer(
                            swe_change_layer=capella_eta_layer_encoded,
                            top_left=top_left,
                            top_right=top_right,
                            bottom_left=bottom_left,
                            bottom_right=bottom_right,
                        ).model_dump_json(),
                    )
                    logger.info("Publishing message successfully completed.")
                    time.sleep(15)


                # Reward
                final_eta_output_file, final_eta_gdf = self.process(
                    gcom_ds=gcom_dataset,
                    snowglobe_ds=capella_dataset,
                    mo_basin=mo_basin,
                    start=old_value,
                    end=new_value,
                )
                # Upload dataset to S3
                self.upload_file(
                    key=final_eta_output_file, filename=final_eta_output_file
                )
                # Clip Final Eta GDF and ground tracks to the Missouri Basin
                final_eta_gdf_clipped = gpd.clip(final_eta_gdf, mo_basin)
                # Convert the clipped GeoDataFrame to GeoJSON and send as message
                all_json_data = final_eta_gdf_clipped.drop(
                    "time", axis=1, errors="ignore"
                ).to_json()
                self.app.send_message(
                    self.app.app_name,
                    "all",
                    VectorLayer(vector_layer=all_json_data).model_dump_json(),
                )
                logger.info("(ALL) Publishing message successfully completed.")
                self.app.send_message(
                    self.app.app_name,
                    "available",
                    VectorLayer(vector_layer=all_json_data).model_dump_json(),
                )

                # Find Optimal Solution#
                output_geojson, selected_cells_gdf = self.find_optimal_solution(
                    final_eta_gdf=final_eta_gdf
                )
                # Upload dataset to S3
                self.upload_file(key=output_geojson, filename=output_geojson)
                selected_cells_gdf["time"] = selected_cells_gdf["time"].astype(str)
                selected_json_data = selected_cells_gdf.to_json()
                self.app.send_message(
                    self.app.app_name,
                    "selected_cells",
                    VectorLayer(vector_layer=selected_json_data).model_dump_json(),
                )
                logger.info("(SELECTED) Publishing message successfully completed.")

    def on_change_alternative(self, source, property_name, old_value, new_value):
        """
        Handle changes to properties

        Args:
            source (object): The source object
            property_name (str): The property name
            old_value (object): The old value
            new_value (object): The new value
        """
        if property_name == "time":
            # Determine if day has changed
            change = self.detect_level_change(new_value, old_value, "day")
            # Publish message if day, week, or month has changed
            if change:
                old_value_reformat = str(old_value.date()).replace("-", "")
                new_value_reformat = str(new_value.date()).replace("-", "")
                # All Tracks
                all_cells_geojson_path = f"Reward_{new_value_reformat}.geojson"
                all_cells_gdf = gpd.read_file(all_cells_geojson_path)
                all_cells_gdf["time"] = all_cells_gdf["time"].astype(str)
                all_json_data = all_cells_gdf.to_json()
                self.app.send_message(
                    self.app.app_name,
                    "all",
                    VectorLayer(vector_layer=all_json_data).model_dump_json(),
                )
                logger.info("(ALL) Publishing message successfully completed.")
                # Selected Cells
                selected_cells_geojson_path = (
                    f"Selected_Cells_Optimization_{new_value_reformat}.geojson"
                )
                selected_cells_gdf = gpd.read_file(selected_cells_geojson_path)
                selected_cells_gdf["time"] = selected_cells_gdf["time"].astype(str)
                selected_json_data = selected_cells_gdf.to_json()
                self.app.send_message(
                    self.app.app_name,
                    "selected_cells",
                    VectorLayer(vector_layer=selected_json_data).model_dump_json(),
                )
                logger.info("(SELECTED) Publishing message successfully completed.")


def main():
    # Load config
    config = ConnectionConfig(yaml_file="sos.yaml")

    # Define the simulation parameters
    NAME = "planner"

    # create the managed application
    app = ManagedApplication(NAME)

    # add the environment observer to monitor simulation for switch to EXECUTING mode
    app.simulator.add_observer(Environment(app))

    # add a shutdown observer to shut down after a single test case
    app.simulator.add_observer(ShutDownObserver(app))

    # start up the application on PREFIX, publish time status every 10 seconds of wallclock time
    app.start_up(
        config.rc.simulation_configuration.execution_parameters.general.prefix,
        config,
        True,
    )

    while True:
        pass


if __name__ == "__main__":
    main()
