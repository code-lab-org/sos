"""
AWS Lambda function to send resume request to NOS-T Manager when data is uploaded to S3.

This Lambda function is triggered when LIS uploads data to a specific S3 location.
It instantiates a NOS-T Application, connects to the event broker, and sends a resume
request to allow the SOS apps to continue execution after an indefinite freeze.
"""

import json
import os
import re
import sys
import time
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from nost_tools.application import Application
from nost_tools.configuration import ConnectionConfig


def get_secret(secret_name, region_name="us-east-1"):
    """
    Retrieve secret from AWS Secrets Manager.

    Args:
        secret_name (str): Name of the secret in Secrets Manager
        region_name (str): AWS region where the secret is stored

    Returns:
        dict: Secret values as a dictionary

    Raises:
        Exception: If secret cannot be retrieved
    """
    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        print(f"Error retrieving secret {secret_name}: {e}")
        raise e

    # Secrets Manager returns the secret as a string, parse it as JSON
    secret = get_secret_value_response["SecretString"]
    return json.loads(secret)


def lambda_handler(event, context):
    """
    Lambda handler triggered by S3 upload events.

    Args:
        event (dict): Lambda event containing S3 upload information
        context: Lambda context object

    Returns:
        dict: Response with status code and message
    """
    app = None

    try:
        s3 = boto3.client("s3")

        # Get bucket and object key from the event
        bucket = event["Records"][0]["s3"]["bucket"]["name"]
        key = event["Records"][0]["s3"]["object"]["key"]
        print(f"Key: {key}")

        # Check if the object is in the "inputs/LIS/assimilation/" folder
        if not key.startswith("inputs/LIS/assimilation/"):
            print(
                f"Object {key} is not in the inputs/LIS/assimilation/ folder, no action taken."
            )
            return {
                "statusCode": 200,
                "body": json.dumps(
                    "Object is not in the inputs/LIS/assimilation/ folder, no action taken."
                ),
            }

        filename = key
        # Extract YYYYMMDD and HHMM
        match = re.search(r"_(\d{8})(\d{4})\.", filename)
        if not match:
            raise ValueError("Filename does not match expected pattern")

        date_part, time_part = match.groups()
        dt = datetime.strptime(f"{date_part}{time_part}", "%Y%m%d%H%M")
        iso_str = dt.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"Iso string: {iso_str}")  # "2025-05-19T00:00:00"

        # Get the time the file was uploaded
        event_time = event["Records"][0]["eventTime"]
        upload_time = datetime.strptime(event_time, "%Y-%m-%dT%H:%M:%S.%fZ")
        upload_time_str = upload_time.strftime("%Y-%m-%d %H:%M:%S")

        print(f"File uploaded: {key} at {upload_time_str}")

        # Get configuration from environment variables
        prefix = os.environ.get("NOST_PREFIX", "nost_sos")
        yaml_config_path = os.environ.get("NOST_CONFIG_YAML", "sos.yaml")
        secret_name = os.environ.get("SECRET_NAME")
        region_name = os.environ.get("AWS_REGION", "us-east-1")

        # Retrieve secrets from AWS Secrets Manager and set as environment variables
        # NOS-T Tools reads USERNAME, PASSWORD, CLIENT_ID, CLIENT_SECRET_KEY from environment
        if secret_name:
            print(f"[DEBUG] Retrieving secrets from AWS Secrets Manager: {secret_name}")
            sys.stdout.flush()
            try:
                print(f"[DEBUG] Calling get_secret with region: {region_name}")
                sys.stdout.flush()
                secrets = get_secret(secret_name, region_name)
                print(f"[DEBUG] Secrets retrieved successfully")
                sys.stdout.flush()

                # # Set environment variables for NOS-T Tools to read
                # if "USERNAME" in secrets:
                #     os.environ["USERNAME"] = secrets["USERNAME"]
                #     print("[DEBUG] Set USERNAME from secrets")
                #     sys.stdout.flush()

                # if "PASSWORD" in secrets:
                #     os.environ["PASSWORD"] = secrets["PASSWORD"]
                #     print("[DEBUG] Set PASSWORD from secrets")
                #     sys.stdout.flush()

                if "CLIENT_ID" in secrets:
                    os.environ["CLIENT_ID"] = secrets["CLIENT_ID"]
                    print("[DEBUG] Set CLIENT_ID from secrets")
                    sys.stdout.flush()

                if "CLIENT_SECRET_KEY" in secrets:
                    os.environ["CLIENT_SECRET_KEY"] = secrets["CLIENT_SECRET_KEY"]
                    print("[DEBUG] Set CLIENT_SECRET_KEY from secrets")
                    sys.stdout.flush()

                print("[DEBUG] Environment variables set from secrets")
                sys.stdout.flush()

            except Exception as e:
                print(f"[ERROR] Failed to retrieve or apply secrets: {e}")
                sys.stdout.flush()
                print(f"[ERROR] Exception type: {type(e).__name__}")
                sys.stdout.flush()
                import traceback

                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                sys.stdout.flush()
                raise e
        else:
            print(
                "[DEBUG] No SECRET_NAME environment variable set, credentials must be in .env file or environment variables"
            )
            sys.stdout.flush()

        # Load configuration - it will automatically read USERNAME, PASSWORD, etc. from environment
        config = ConnectionConfig(yaml_file=yaml_config_path)
        print(f"Configuration loaded successfully from: {yaml_config_path}.")

        # Create the Application instance
        app = Application(app_name="lambda_trigger")
        print(f"Application {app.app_name} successfully started.")

        # Start up the application (connect to RabbitMQ/Keycloak)
        app.start_up(
            config.rc.simulation_configuration.execution_parameters.general.prefix,
            config,
            True,
        )

        # Resume
        app.request_resume(
            sim_resume_time=iso_str,
            # tolerance=timedelta(seconds=5),
        )

        print(f"Resume request sent successfully for file: {key}")

        # Give a moment for the message to be sent
        time.sleep(1)
        return {
            "statusCode": 200,
            "message": "Data availability nessage sent successfully.",
        }

    except Exception as e:
        print(f"Error processing event: {e}")
        import traceback

        print(f"Traceback: {traceback.format_exc()}")
        return {"statusCode": 500, "body": json.dumps(f"Error: {str(e)}")}
