import os
import re
from os.path import exists

from google.cloud import storage


# Need to save service account JSON under root_repo_name directory


class GCP:
    def __init__(
        self,
        service_account_name="omd-emea-daimler-a8c6772fb5f7.json",
    ):
        """
        Args:
            service_account_name (str, optional): file name of the Service account JSON file which is stored directly in the root_repo directory   eg. 'omd-emea-daimler-a8c6772fb5f7.json'
        """

        self.json_path = service_account_name
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_name

    def download_file_from_bucket(self, GCP_path, destination_file_name):
        """Downloads file from bucket

        Args:
            GCP_path (str): The path to the file in GCP
            destination_file_name (str): relative path where to download file into
        """

        bucket_name, source_blob_name = GCP_path.split("/", 1)

        storage_client = storage.Client.from_service_account_json(self.json_path)
        bucket = storage_client.bucket(bucket_name)

        blob = bucket.blob(source_blob_name)
        blob.download_to_filename(destination_file_name)

        print(
            "Downloaded storage object {} from bucket {} to local file {}.".format(
                source_blob_name, bucket_name, destination_file_name
            )
        )

    def __check_variable_names_have_been_set__(self, varible_names, varible_values):
        d = dict(zip(varible_names, varible_values))
        for var_name in d:
            if d[var_name] == None:
                raise ValueError(f"GCP: {var_name} not defined")

    def upload_to_bucket(self, GCP_path, relative_path_to_file):
        """Uploads Data to Bucket

        Args:
            GCP_path (str): The path to the file in GCP
            relative_path_to_file (str): Relative path to locally stored file

        Returns
            str: Location of stored file in GCP bucket
        """

        bucket_name, blob_name = GCP_path.split("/", 1)

        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(relative_path_to_file)

        # returns a public url
        return blob.public_url

    def search_for_file_name(self, gcp_bucket, file):
        """Searches a GCP bucket for a file, and returns path in GCP bucket if it exists

        Args:
            gcp_bucket (str): Bucket to search in
            file (str): file to look for
        """

        list_of_matching_files = (
            os.popen('gsutil ls "gs://' + gcp_bucket + '/**" | grep ' + file)
            .read()
            .split("\n")[:-1]
        )

        return list_of_matching_files

    def download_multiple_files(self, GCP_path, regex_pattern, relative_path_to_folder):
        """Downloads multiple files under a path in GCP according to a regex match, and stores them all in one folder

        Args:
            GCP_path (str): Folder in GCP you wish to search for multiple files
            regex_pattern (str): regex pattern matching criteria
            relative_path_to_folder (str): path to local folder to store all downloaded files into, if you are installing directly in root repo dir then put ''
        """
        # add check if files are called same thing

        files_to_download = self.search_for_file_name(GCP_path, regex_pattern)

        try:
            bucket_name, _ = GCP_path.split("/", 1)
        except:
            # TODO make this more granular
            # If we are looking directly in root of the bucket
            bucket_name = GCP_path
        storage_client = storage.Client.from_service_account_json(self.json_path)
        bucket = storage_client.bucket(bucket_name)

        for file in files_to_download:
            file = file.replace("gs://" + bucket_name + "/", "")
            file_name = file.split("/")[-1]

            blob = bucket.blob(file)
            blob.download_to_filename(os.path.join(relative_path_to_folder, file_name))

    def read_in_data(
        self, download_from_GCP, relative_path_to_file, file_reader, GCP_path=None
    ):
        """Reads in data from existing local file or downloads from GCP to path_to_file and reads it in

        Args:
            download_from_GCP (bool): Whether or not to download file from GCP or use locally stored file to read
            relative_path_to_file (str): relative path to locally stored file / where to download file into
            file_reader (str): method to read file
            GCP_path (str): The path to the file in GCP

        Returns:
            data: The data eg. pandas df
        """
        # TODO add **kwargs for file_reader

        if download_from_GCP:
            try:
                self.__check_variable_names_have_been_set__(["GCP_path"], [GCP_path])
            except Exception as e:
                return e
            self.download_file_from_bucket(GCP_path, relative_path_to_file)

        else:
            if not exists(local_path_to_file):
                print(
                    "Missing data, please read in from GCP or enter correct data location"
                )

        return file_reader(relative_path_to_file)
