import os


class DataUtils(object):
    """
    A class for working with application directories and resources.
    """

    def __init__(self, app_name=None):
        """
        Initialize the class with optional application name.

        Args:
            app_name (str, optional): Application name for app-specific directories
        """
        self.app_name = app_name
        self.dirs = {}

    def create_directories(self, directory_list, parent_dir=None):
        """
        Create directories and store their paths.

        Args:
            directory_list (list): List of directories to create
            parent_dir (str, optional): Parent directory for these directories

        Returns:
            dict: Dictionary of created directory paths
        """
        for directory in directory_list:
            # Build the full path
            if self.app_name and parent_dir:
                full_path = os.path.join(parent_dir, self.app_name, directory)
            elif self.app_name:
                full_path = os.path.join(self.app_name, directory)
            elif parent_dir:
                full_path = os.path.join(parent_dir, directory)
            else:
                full_path = directory

            # Create the directory
            os.makedirs(full_path, exist_ok=True)

            # Store the path
            self.dirs[directory] = full_path

        return self.dirs

    def get_path(self, directory):
        """
        Get the path for a specific directory.

        Args:
            directory (str): Directory name

        Returns:
            str: Full path to the directory
        """
        return self.dirs.get(directory)


# import os

# class AppUtils(object):
#     """
#     A class for working with applications.
#     """

#     def __init__(self):
#         """
#         Initialize the class.
#         """
#         self.create_directories(["input", "output"])

#     def create_directories(self, directory_list):
#         """
#         Create the input and output directories.

#         Input:
#             directory_list (list): List of directories to create
#         """
#         for directory in directory_list:
#             os.makedirs(directory, exist_ok=True)
