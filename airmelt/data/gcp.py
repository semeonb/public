import os
import shutil
import random
import string
import glob
import tensorflow as tf
import joblib
import json
import logging

from google.cloud import bigquery
from google.cloud import storage

from airmelt_system import toolbox
from airmelt_data import aws


CSV = "CSV"
WRITE_APPEND = "WRITE_APPEND"
WRITE_TRUNCATE = "WRITE_TRUNCATE"
CREATE_IF_NEEDED = "CREATE_IF_NEEDED"
NEWLINE_DELIMITED_JSON = "NEWLINE_DELIMITED_JSON"


def create_local_dir(dir_path, restart=False):
    """
    dir_path: the path of the directory to be created
    restart: a boolean flag that determines whether the directory should be deleted and recreated if it already exists.
    """
    try:
        if os.path.isdir(dir_path):
            if restart:
                shutil.rmtree(dir_path)
                os.makedirs(dir_path)
        else:
            os.makedirs(dir_path)
    except OSError as e:
        print("Error creating directory:", e)


def delete_local_folder(folder):
    """
    folder: the path of the folder to be deleted
    """
    try:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
    except OSError as e:
        print("Error deleting folder:", e)


def get_schema(schemaFile):
    """
    schemaFile: the path of the JSON file containing the schema.
    """
    try:
        with open(schemaFile) as schema_data:
            obj = json.load(schema_data)
            for i in obj:
                if "description" not in i:
                    i["description"] = ""
            return obj
    except FileNotFoundError as e:
        print(f"Error: {schemaFile} not found")
        raise e
    except Exception as e:
        print(f"Error reading {schemaFile}")
        raise e
    finally:
        schema_data.close()


# Generates random string of a given length
def generate_random_string(length):
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))


class Bigquery(object):
    def __init__(
        self, bq_project_id: str, gs_project_id=None, intermediate_dataset="tmp"
    ):
        """
        BigQuery class
        bq_project_id: Name of BQ project
        gs_project_id: Name of GS project
        intermediate_dataset: Name of the Dataset to be used as a staging area
        """
        self.bq_project_id = bq_project_id
        self.intermediate_dataset = intermediate_dataset
        self.bq_client = bigquery.Client(project=self.bq_project_id)
        if not gs_project_id:
            self.gs_project_id = bq_project_id
        else:
            self.gs_project_id = gs_project_id
        self.gs_client = storage.Client(project=self.gs_project_id)
        self.logger = logging.getLogger(__name__)

    def execute_query(
        self,
        query: str,
        destination_table=None,
        destination_dataset=None,
        schema=None,
        write_disposition="WRITE_APPEND",
        create_disposition="CREATE_IF_NEEDED",
    ):
        """
        Executes a bigquery SQL query and returns the result.
        query: str: SQL query string
        """
        # create a query job configuration
        job_config = bigquery.QueryJobConfig()
        # use standard SQL
        job_config.use_legacy_sql = False
        # start the query job
        query_job = self.bq_client.query(query, job_config=job_config)
        # wait for the query to finish and return the results
        if not destination_table:
            return query_job.result()
        else:
            return self.query2table(
                query,
                destination_table,
                destination_dataset,
                create_disposition,
                write_disposition,
                schema,
            )

    def query_to_list(self, query):
        """
        Creates de-serialized BQ result dataset
        query: query string
        batch_size: will create a list of lists with a specified length
        """
        return [
            json.dumps(dict(row), default=toolbox.default_dict_converter)
            for row in self.execute_query(query)
        ]

    def query2vector(self, query: str):
        """
        Return query result in form of a list
        query: query string
        """
        vector = []
        x = self.execute_query(query=query)
        for row in x:
            for i in row:
                vector.append(i)
        return vector

    def query2table(
        self,
        query: str,
        dest_table: str,
        dest_dataset: str,
        create_disposition=CREATE_IF_NEEDED,
        write_disposition=WRITE_APPEND,
        schema=None,
    ):
        """
        Saves query results to BQ table
        query: query string
        dest_table: destination table
        dest_dataset: destination dataset
        create_disposition: Behavior in case the table exists
        write_disposition: How to write the data in (TRUNCATE/APPEND)
        """
        job_config = bigquery.QueryJobConfig()
        if schema:
            job_config.schema = schema
        job_config.create_disposition = create_disposition
        job_config.write_disposition = write_disposition
        table_ref = self.bq_client.dataset(dest_dataset).table(dest_table)
        job_config.destination = table_ref
        query_job = self.bq_client.query(query, job_config=job_config)
        return query_job.result()

    def bqtable2gs(
        self,
        dataset_name,
        table_name,
        bucket,
        path,
        file_name,
        compression=False,
        destination_format="NEWLINE_DELIMITED_JSON",
        print_header=True,
        gzip_extension=".gzip",
    ):
        # this function exports a bigquery table to a google storage bucket
        # dataset_name: name of the bigquery dataset
        # table_name: name of the table to be exported
        # bucket: name of the google storage bucket
        # path: path within the bucket where the file should be saved
        # file_name: name of the file to be saved
        # compression: whether to compress the file or not (default: False)

        # job configuration for the bigquery export
        job_config = bigquery.ExtractJobConfig()

        # reference to the dataset
        dataset_ref = self.bq_client.dataset(dataset_name)

        # set the destination format
        job_config.destination_format = destination_format

        job_config.print_header = print_header

        if destination_format == NEWLINE_DELIMITED_JSON:
            file_name = file_name + ".json"
        else:
            file_name = file_name + ".csv"

        # if compression is set to True, use GZIP compression
        if compression:
            job_config.compression = "GZIP"
            file_name = file_name + gzip_extension

        # destination URI for the file
        destination_uri = "gs://" + os.path.join(bucket, path, file_name)

        # start the export job
        extract_job = self.bq_client.extract_table(
            dataset_ref.table(table_name), destination_uri, job_config=job_config
        )
        return extract_job.result(), destination_uri

    def localfile2gs(self, bucket, source_path, dest_path, delete_local=False):
        # create a client object for Google Cloud Storage
        client = storage.Client(project=self.gs_project_id)

        # get a reference to the specified bucket
        bucketObj = client.get_bucket(bucket)

        # create a new blob object in the bucket
        blob = bucketObj.blob(dest_path)

        # upload the local file to the bucket
        blob.upload_from_filename(source_path)

        # check if local file should be deleted
        if delete_local:
            # remove the local file
            os.remove(source_path)

        # return the URL of the uploaded file
        return "gs://" + os.path.join(bucket, dest_path)

    def gs2localfile(self, bucket, blob, dest_path):
        # split the blob path to get the file name
        _, file = os.path.split(blob)

        # create a reference to the specified bucket
        bucket = self.gs_client.get_bucket(bucket)

        # create a new blob object in the bucket
        blob = bucket.blob(blob)

        full_filename = os.path.join(dest_path, file)

        # download the file from the bucket to the local directory
        try:
            blob.download_to_filename(full_filename)
        except Exception as e:
            print("Could  not download file: " + str(e) + " - " + full_filename)
            raise

        # return the local path of the downloaded file
        return full_filename

    def list_blobs(self, bucket_name, path, obj_output=False):
        """
        bucket_name: the name of the GS bucket that we want to list the files from.
        path: the prefix path we want to filter by, to list only the files under that specific path.
        obj_output: if set to True, the function will return a list of blob objects instead of a list of file names.
        """
        # create an empty list to store the names of the files found
        objects = list()

        # create a storage client object to interact with the Google Cloud Storage service
        storage_client = storage.Client()

        # get a reference to the specified bucket
        bucket = storage_client.get_bucket(bucket_name)

        # get all the blobs under the specified prefix path
        blobs = bucket.list_blobs(prefix=path)

        # loop through the blobs
        for blob in blobs:
            # append the blob's name / object to the objects list
            if obj_output:
                objects.append(blob)
            else:
                objects.append(blob.name)

        # return the objects list containing the names of the files found
        return objects

    def copy_gs_folder_to_local(self, bucket_name, source_path, dest_path):
        blobs = self.list_blobs(bucket_name, source_path)
        # take only the last folder path section
        prefix = "/".join(source_path.split("/")[:-1]) + "/"
        for b in blobs:
            # separate file from folder
            folder, file = os.path.split(b)
            if folder.startswith(prefix):
                folder = folder[len(prefix) :]
            os.makedirs(os.path.join(dest_path, folder), exist_ok=True)
            # check if the blob is a folder
            if file == "":
                continue
            else:
                bucket = self.gs_client.get_bucket(bucket_name)
                blob = bucket.blob(b)
                blob.download_to_filename(os.path.join(dest_path, folder, file))

    def delete_blobs(self, bucket_name, path):
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)

        blobs = bucket.list_blobs(prefix=path)
        for blob in blobs:
            blob.delete()

    def delete_blob(self, bucket_name, blob_name):
        """Deletes a blob from the bucket."""
        # bucket_name = "your-bucket-name"
        # blob_name = "your-object-name"

        storage_client = storage.Client()

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        generation_match_precondition = None

        # Optional: set a generation-match precondition to avoid potential race conditions
        # and data corruptions. The request to delete is aborted if the object's
        # generation number does not match your precondition.
        blob.reload()  # Fetch blob metadata to use in generation_match_precondition.
        generation_match_precondition = blob.generation

        blob.delete(if_generation_match=generation_match_precondition)

        self.logger.info(f"Blob {blob_name} deleted.")

    def load(
        self,
        dataset,
        table,
        path,
        source_format,
        schema=None,
        bad_record_limit=0,
        skip_leading_rows=0,
        write_flag=WRITE_APPEND,
        ad_flag=False,
        allow_jagged_rows=False,
        create_disposition=CREATE_IF_NEEDED,
        allow_quoted_newlines=False,
        ignore_unknown_values=False,
        field_delimiter=",",
        quote_character='"',
    ):

        """
        dataset: the name of the BigQuery dataset in which the table is located
        table: the name of the BigQuery table to which the data will be loaded
        path: the path of the file to be loaded
        source_format: the format of the file (CSV or JSON)
        schema: the schema for the table, if provided
        bad_record_limit: the maximum number of bad records that are allowed before the job fails
        skip_leading_rows: the number of rows to skip at the beginning of the file (only used for CSV files)
        write_flag: the write disposition for the data (options are WRITE_TRUNCATE, WRITE_APPEND, WRITE_EMPTY)
        ad_flag: a flag indicating whether or not the schema should be automatically detected
        allow_jagged_rows: a flag indicating whether or not the table can contain rows with varying numbers of columns
        create_disposition: the create disposition for the table (options are CREATE_IF_NEEDED, CREATE_NEVER)
        allow_quoted_newlines: a flag indicating whether or not newlines are allowed in quoted strings
        ignore_unknown_values: a flag indicating whether or not unknown values should be ignored
        field_delimiter: the delimiter used to separate fields in the file (only used for CSV files)
        """

        # create a dataset reference using the bq_client and the specified dataset name
        dataset_ref = self.bq_client.dataset(dataset)

        # create a LoadJobConfig object
        job_config = bigquery.LoadJobConfig()

        # set the maximum number of bad records
        job_config.max_bad_records = bad_record_limit

        # if a schema is provided, assigns it to the schema attribute of the job config
        if schema:
            job_config.schema = schema
        # set source_format
        job_config.source_format = source_format

        # set the autodetect flag
        job_config.autodetect = ad_flag

        # set the allow_jagged_rows flag
        job_config.allow_jagged_rows = allow_jagged_rows

        # set the ignore_unknown_values flag
        job_config.ignore_unknown_values = ignore_unknown_values

        #  set the allow_quoted_newlines flag
        job_config.allow_quoted_newlines = allow_quoted_newlines

        # set the write disposition
        job_config.write_disposition = write_flag

        # set the create disposition
        job_config.create_disposition = create_disposition
        if source_format == CSV:

            # set the field delimiter
            job_config.field_delimiter = field_delimiter

            # set the number of rows to skip at the beginning of the file
            job_config.skip_leading_rows = skip_leading_rows

            # set the quote character
            job_config.quote_character = quote_character

        # start the job to load the data into the table
        load_job = self.bq_client.load_table_from_uri(
            path, dataset_ref.table(table), job_config=job_config
        )

        return load_job.result()

    def localtable2bq(
        self,
        bucket_name,
        dataset,
        table,
        skip_leading_rows,
        write_flag,
        ad_flag,
        source_path,
        source_format,
        schema=None,
        intermediatePath="tmp",
        field_delimiter=",",
        ignore_unknown_values=True,
        allow_jagged_rows=True,
        allow_quoted_newlines=True,
        quote_character='"',
        bad_record_limit=0,
    ):
        _, filename = os.path.split(source_path)
        # load file to intermediate path
        self.localfile2gs(
            bucket=bucket_name,
            source_path=source_path,
            dest_path=os.path.join(intermediatePath, filename),
        )
        # load file from GS to BQ
        path = "gs://" + os.path.join(bucket_name, intermediatePath, filename)
        self.load(
            dataset=dataset,
            table=table,
            path=path,
            source_format=source_format,
            skip_leading_rows=skip_leading_rows,
            write_flag=write_flag,
            ad_flag=ad_flag,
            schema=schema,
            field_delimiter=field_delimiter,
            ignore_unknown_values=ignore_unknown_values,
            allow_jagged_rows=allow_jagged_rows,
            allow_quoted_newlines=allow_quoted_newlines,
            quote_character=quote_character,
            bad_record_limit=bad_record_limit,
        )

        # delete GS file
        try:
            self.delete_blob(
                bucket_name=bucket_name,
                blob_name=os.path.join(intermediatePath, filename),
            )
            self.logger.info(
                "File {} deleted from GS".format(
                    os.path.join(intermediatePath, filename)
                )
            )
        except Exception as e:
            self.logger.error("Error deleting file from GS: {}".format(e))

    def delete_bq_table(self, dataset, table):
        if self.checkTableExists(dataset, table):
            table_ref = self.bq_client.dataset(dataset).table(table)
            self.bq_client.delete_table(table_ref)
            return "Table deleted"
        else:
            return "Table does not exist"

    def export_query_to_gcs(
        self,
        query,
        bucket,
        temp_dataset,
        temp_table,
        path,
        file_name,
        destination_format=CSV,
        compression=False,
        delete_after_export=True,
        gzip_extension=".gzip",
    ):

        self.query2table(
            query=query,
            dest_table=temp_table,
            dest_dataset=temp_dataset,
            create_disposition=CREATE_IF_NEEDED,
            write_disposition=WRITE_TRUNCATE,
        )
        result, destination = self.bqtable2gs(
            dataset_name=temp_dataset,
            table_name=temp_table,
            bucket=bucket,
            path=path,
            file_name=file_name,
            compression=compression,
            destination_format=destination_format,
            gzip_extension=gzip_extension,
        )
        if delete_after_export:
            self.delete_bq_table(dataset=temp_dataset, table=temp_table)

        return result, destination

    def export_query_to_csv(
        self,
        query,
        temp_table,
        temp_filename,
        temp_dataset,
        temp_bucket,
        temp_path="data",
        split_files=True,
        delete_after_export=True,
        intermediatePath="tmp",
    ):
        file_name, file_extension = os.path.splitext(temp_filename)
        if not file_extension:
            file_extension = ".csv"
        if split_files:
            path = os.path.join(temp_path, file_name)
            gs_file_name = file_name + "_*" + file_extension
        else:
            path = temp_path
            gs_file_name = file_name + file_extension
        create_local_dir(dir_path=path)
        self.query2table(
            self,
            query=query,
            dest_table=temp_table,
            dest_dataset=temp_dataset,
            create_disposition=CREATE_IF_NEEDED,
            write_disposition=WRITE_TRUNCATE,
        )
        # clean intermediate GS directory
        gs_path = os.path.join(intermediatePath, file_name)
        try:
            self.delete_blobs(self, bucket_name=temp_bucket, path=gs_path)
        except Exception:
            pass
        self.bqtable2gs(
            self,
            dataset_name=temp_dataset,
            table_name=temp_table,
            bucket=temp_bucket,
            path=gs_path,
            file_name=gs_file_name,
            compression=False,
        )
        if delete_after_export:
            self.delete_bq_table(self, dataset=temp_dataset, table=temp_table)
        # Get list of files in GS
        blobs = self.list_blobs(self, bucket_name=temp_bucket, path=gs_path)
        # create local directory if needed
        files = []
        for blob in blobs:
            file = self.gs2localfile(
                self,
                bucket=temp_bucket,
                blob=blob,
                dest_path=path,
            )
            files.append(file)
        # delete blobs
        self.delete_blobs(self, bucket_name=temp_bucket, path=gs_path)

        return files, path

    def query2list(self, query):
        query_job = self.bq_client.query(query)
        rows = query_job.result()
        result = []
        for row in rows:
            result.append([i for i in row])
        schema = [i.name for i in rows.schema]
        return result, schema

    def query_to_dataframe(self, query):
        return self.bq_client.query(query).to_dataframe()

    def get_table_suffix_ends(self, dataset, schema_name):
        """ "
        the method returns the first and last date of the table
        """
        query = """
            SELECT 
            MIN(DATE(parse_timestamp('%Y%m%d',_TABLE_SUFFIX))) AS mindate,
            MAX(DATE(parse_timestamp('%Y%m%d',_TABLE_SUFFIX))) AS maxdate
            FROM `{ds}.{schema}*`
            """.format(
            ds=dataset, schema=schema_name
        )
        try:
            suffix_ends = self.query2vector(query=query)
        except Exception as ex:
            self.logger.error(str(ex))
            suffix_ends = None
        return suffix_ends

    def get_partitioned_table_ends(self, dataset, schema_name, field="_PARTITIONTIME"):
        query = """
                    SELECT 
                    MIN(DATE({field})) AS mindate, 
                    MAX(DATE({field})) AS maxdate 
                    FROM `{ds}.{schema}`
                """.format(
            ds=dataset, schema=schema_name, field=field
        )
        return self.query2vector(query=query)

    def list_tables(self, dataset):
        dataset_ref = self.bq_client.dataset(dataset)
        if not dataset_ref:
            print("Dataset {} does not exist.".format(dataset))
            return None
        else:
            return [i.table_id for i in list(self.bq_client.list_tables(dataset_ref))]

    def checkTableExists(self, dataset, table):
        if table in self.list_tables(dataset=dataset):
            return True
        else:
            return False

    def create_table(
        self,
        datasetName,
        tableName,
        schema,
        partition_col_name=None,
        expirtion_days=None,
    ):
        if expirtion_days:
            expiration_ms = 86400000 * expirtion_days
        else:
            expiration_ms = None
        dataset_ref = self.bq_client.dataset(datasetName)
        table_ref = dataset_ref.table(tableName)
        table = bigquery.Table(table_ref, schema=schema)
        if partition_col_name:
            table.time_partitioning = bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=partition_col_name,
                expiration_ms=expiration_ms,
            )
        table = self.bq_client.create_table(table)
        return table

    def extract_table_schema(self, dataset, table_name):
        """
        This function extracts the schema of a BigQuery table
        """
        # Construct a BigQuery client object.
        dataset_ref = self.bq_client.dataset(dataset)
        table_ref = dataset_ref.table(table_name)
        table = self.bq_client.get_table(table_ref)
        return table.schema

    def generate_schema_from_json(self, json_object):
        schema = []
        for i in json_object:
            if "description" not in i:
                i["description"] = ""
            if "mode" not in i:
                i["mode"] = "NULLABLE"
            if not i.get("type") == "RECORD":
                schema.append(
                    bigquery.SchemaField(
                        i.get("name"),
                        i.get("type"),
                        i.get("mode"),
                        i.get("description"),
                    )
                )
            else:
                nested_schema = []
                for k in i["fields"]:
                    if "description" not in k:
                        k["description"] = ""
                    nested_schema.append(
                        bigquery.SchemaField(
                            k.get("name"),
                            k.get("type"),
                            k.get("mode"),
                            k.get("description"),
                        )
                    )
                schema.append(
                    bigquery.SchemaField(
                        i.get("name"),
                        i.get("type"),
                        i.get("mode"),
                        i.get("description"),
                        (nested_schema),
                    )
                )
        return schema

    def generate_schema_from_file(self, schemaFile):
        json_object = get_schema(schemaFile)
        schema = self.generate_schema_from_json(json_object)
        return schema

    def copy_bq_table(
        self,
        source_dataset,
        source_table,
        target_dataset,
        target_table,
        write_disposition="WRITE_TRUNCATE",
    ):
        """
        source_dataset: the name of the BigQuery dataset that contains the source table
        source_table: the name of the source table
        target_dataset: the name of the BigQuery dataset that contains the target table
        target_table: the name of the target table
        write_disposition: the write disposition for the data (options are WRITE_TRUNCATE, WRITE_APPEND, WRITE_EMPTY)
        """

        # create a CopyJobConfig object
        job_config = bigquery.CopyJobConfig()

        # set the write disposition
        job_config.write_disposition = write_disposition

        # create a reference to the source dataset
        source_dataset = self.bq_client.dataset(source_dataset)

        # create a reference to the source table
        source_table_ref = source_dataset.table(source_table)

        # create a reference to the target table
        dest_table_ref = self.bq_client.dataset(target_dataset).table(target_table)

        # initiate the job of copying the source table to the target table
        job = self.bq_client.copy_table(
            source_table_ref, dest_table_ref, job_config=job_config
        )
        # wait for the job to complete
        result = job.result()

        # assert that the job state is "DONE"
        assert job.state == "DONE"
        dest_table = self.bq_client.get_table(dest_table_ref)

        # assert that the target table has more than 0 rows
        assert dest_table.num_rows > 0

        return result

    def upload_from_directory(
        self, directory_path: str, dest_bucket_name: str, dest_blob_name: str
    ):
        """
            The method copies all files recursively from a local folder to remote GS folder
            upload_from_directory(
            directory_path="/tmp/conversion_event/temporary_models/day_7/binary",
            dest_bucket_name="GS bucket name",
            dest_blob_name="am_models/conversion_event/temporary_models/day_7/binary",
        )
        """
        rel_paths = glob.glob(directory_path + "/**", recursive=True)
        current_folder_position = len(directory_path.split(os.sep))
        bucket = self.gs_client.get_bucket(dest_bucket_name)
        for local_file in rel_paths:
            local_file_list = local_file.split(os.sep)
            suffix = local_file_list[current_folder_position:]
            remote_path = os.path.join(dest_blob_name, os.sep.join(suffix))
            if os.path.isfile(local_file):
                print(remote_path)
                blob = bucket.blob(remote_path)
                blob.upload_from_filename(local_file)

    def copy_blob(
        self, bucket_name, blob_name, destination_bucket_name, destination_blob_name
    ):
        """Copies a blob from one bucket to another with a new name."""
        # bucket_name = "your-bucket-name"
        # blob_name = "your-object-name"
        # destination_bucket_name = "destination-bucket-name"
        # destination_blob_name = "destination-object-name"

        source_bucket = self.gs_client.bucket(bucket_name)
        source_blob = source_bucket.blob(blob_name)
        destination_bucket = self.gs_client.bucket(destination_bucket_name)

        source_bucket.copy_blob(source_blob, destination_bucket, destination_blob_name)

    def query_to_df(self, sql):
        return self.bq_client.query(sql).to_dataframe()

    def read_from_gs(self, bucket_name, blob_name, download_as="bytes"):
        """
        The method reads a file from GS and returns it as bytes or string
        """
        bucket = self.gs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        if download_as == "bytes":
            return blob.download_as_bytes()
        elif download_as == "string":
            return blob.download_as_string
        else:
            raise Exception("unknown download type")

    def read_gs_model(self, bucket_name, blob_name):
        gcs_path = os.path.join(bucket_name, blob_name)
        return joblib.load(tf.io.gfile.GFile("gs://" + gcs_path, "rb"))

    def upload_to_s3(
        self,
        source_bucket,
        destination_bucket,
        source_blob,
        destination_folder,
        aws_access_key_id,
        aws_secret_access_key,
        aws_region_name,
        intermediateFolder="/tmp",
    ):
        """
        The method copies a file from GS to S3
        """

        self.logger.info("Downloading file from GS to local")
        try:
            local_file = self.gs2localfile(
                source_bucket, source_blob, intermediateFolder
            )
        except Exception as e:
            self.logger.error("Error downloading file from GS to local")
            raise e

        # getting s3 session
        s3_session = aws.S3(aws_access_key_id, aws_secret_access_key, aws_region_name)
        try:
            self.logger.info("Uploading file to S3")
            s3_session.upload_file(local_file, destination_bucket, destination_folder)
        except Exception as e:
            self.logger.error("Error uploading file to S3")
            raise e

        # deleting local file
        try:
            self.logger.info("Deleting local file")
            toolbox.delete_local_file(local_file)
        except Exception as e:
            self.logger.error("Error deleting local file")
        return os.path.basename(source_blob)

    def df2bq(
        self,
        dataset,
        table,
        df,
        bucket_name,
        skip_leading_rows=None,
        write_flag=WRITE_TRUNCATE,
        ad_flag=False,
        schema=None,
        ignore_unknown_values=True,
        allow_jagged_rows=True,
        work_dir="/tmp",
        bad_record_limit=0,
    ):
        """
        This method is used to upload a dataframe to a BigQuery table
        """

        # create a temporary file with a random name
        filename = toolbox.generate_random_string(8) + ".json"
        # create the full path to the temporary file
        filepath = os.path.join(work_dir, filename)
        # write the dataframe to the temporary file
        df.to_json(filepath, orient="records", lines=True)
        try:
            # upload the temporary file to GS
            self.localtable2bq(
                dataset=dataset,
                table=table,
                bucket_name=bucket_name,
                skip_leading_rows=skip_leading_rows,
                write_flag=write_flag,
                ad_flag=ad_flag,
                source_path=filepath,
                source_format=NEWLINE_DELIMITED_JSON,
                schema=schema,
                ignore_unknown_values=ignore_unknown_values,
                allow_jagged_rows=allow_jagged_rows,
                field_delimiter=None,
                bad_record_limit=bad_record_limit,
            )
            # delete the temporary file
            toolbox.delete_local_file(filepath)
        except Exception:
            toolbox.delete_local_file(filepath)
            raise

    def delete_rows_from_table(self, dataset_name, table_name):
        """
        Deletes all rows from a BigQuery table.

        Args:
            dataset_name (str): The name of the dataset containing the table.
            table_name (str): The name of the table to delete rows from.
            bq_project_id (str): The ID of the BigQuery project.
            gcs_project_id (str): The ID of the GCS project.

        Returns:
            None
        """
        table_exists = self.checkTableExists(dataset=dataset_name, table=table_name)
        if not table_exists:
            raise ValueError(
                f"Table {table_name} does not exist in dataset {dataset_name}"
            )

        query = f"DELETE FROM `{dataset_name}`.`{table_name}` WHERE 1 = 1"
        return self.execute_query(query=query)
