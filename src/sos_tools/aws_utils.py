import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config


class AWSUtils(object):
    """
    A class for working with AWS.
    """

    def __init__(self, mfa_serial=None, mfa_token=None, mfa_required=False):
        """
        Initialize the class.

        Args:
            mfa_serial (str): MFA serial number
            mfa_token (str): MFA token
            mfa_required (bool): If True, MFA is required
        """
        self.mfa_serial = mfa_serial
        self.mfa_token = mfa_token
        self.mfa_required = mfa_required
        self.session = self.start_session()
        self.token = self.get_session_token()
        self.client = self.create_client()

    def start_session(self):
        """
        Start a boto3 session.

        Returns:
            session (boto3.Session): A boto3 session
        """
        session = boto3.Session()
        return session

    def get_session_token(self):
        """
        Get a session token.

        Returns:
            token (dict): A dictionary containing the session token
        """
        sts = self.session.client("sts")
        if self.mfa_required:
            return sts.get_session_token(
                SerialNumber=self.mfa_serial, TokenCode=self.mfa_token
            )
        else:
            return sts.get_session_token()

    def decompose_token(self):
        """
        Decompose the session token.

        Returns:
            session_token (str): The session token
            secret_access_key (str): The secret access key
            access_key_id (str): The access key ID
        """
        credentials = self.token.get("Credentials", {})
        session_token = credentials.get("SessionToken")
        secret_access_key = credentials.get("SecretAccessKey")
        access_key_id = credentials.get("AccessKeyId")
        return session_token, secret_access_key, access_key_id

    def create_client(self):
        """
        Create an AWS client with increased connection pool size to prevent
        "Connection pool is full" warnings.

        Returns:
            client (boto3.client): An AWS client
        """
        session_token, secret_access_key, access_key_id = self.decompose_token()

        # Configure client with larger connection pool to handle concurrent requests
        config = Config(
            max_pool_connections=50,  # Increase from default of 10
            retries={"max_attempts": 3, "mode": "adaptive"},
        )

        client = boto3.client(
            "s3",
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token,
            config=config,
        )
        return client

    def upload_file(self, s3, bucket, key, filename):
        """
        Upload a file to an S3 bucket

        Args:
            s3: S3 client
            bucket: S3 bucket name
            key: S3 object key
            filename: Filename to upload
        """
        # logger.info(f"Uploading file to S3.")
        config = TransferConfig(use_threads=False)
        s3.upload_file(Filename=filename, Bucket=bucket, Key=key, Config=config)
        # logger.info(f"Uploading file to S3 successfully completed.")


def start_session():
    """
    Start a boto3 session.

    Returns:
        session (boto3.Session): A boto3 session
    """
    session = boto3.Session()
    return session


def get_session_token(session, mfa_serial=None, mfa_token=None, mfa_required=True):
    """
    Get a session token.

    Args:
        session (boto3.Session): A boto3 session
        mfa_serial (str): MFA serial number
        mfa_token (str): MFA token
        mfa_required (bool): If True, MFA is required

    Returns:
        token (dict): A dictionary containing the session token
    """
    sts = session.client("sts")
    if mfa_required:
        return sts.get_session_token(SerialNumber=mfa_serial, TokenCode=mfa_token)
    else:
        return sts.get_session_token()


def decompose_token(token):
    """
    Decompose the session token.

    Args:
        token (dict): A dictionary containing the session token

    Returns:
        session_token (str): The session token
        secret_access_key (str): The secret access key
        access_key_id (str): The access key ID
    """
    credentials = token.get("Credentials", {})
    session_token = credentials.get("SessionToken")
    secret_access_key = credentials.get("SecretAccessKey")
    access_key_id = credentials.get("AccessKeyId")
    return session_token, secret_access_key, access_key_id
