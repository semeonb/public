#!/usr/bin/env python3
from os import path

from google.cloud import bigquery
from google.cloud import storage


CREATE_IF_NEEDED = 'CREATE_IF_NEEDED'
WRITE_APPEND = 'WRITE_APPEND'
CSV = 'CSV'


def genSchemaFromDict(schemaDict):

    def bqSchemaObj(schemaObj):
        fields = []
        if 'description' not in schemaObj:
            schemaObj['description'] = ""
        if schemaObj['mode'] == 'REPEATED':
            for k in schemaObj['fields']:
                fields.append(bqSchemaObj(k))

        return bigquery.SchemaField(schemaObj['name'], schemaObj['type'],
                                    schemaObj['mode'], schemaObj['description'], fields)

    schema = []
    for i in schemaDict:
        schema.append(bqSchemaObj(i))
    return schema


class GoogleCloud(object):

    def __init__(self, project_id):
        self.project_id = project_id

    def client(self, project_id):
        return bigquery.Client(project=project_id)

    def createTable(self, table, dataset, schema):
        bq_client = self.client(self.project_id)
        table_ref = bq_client.dataset(dataset).table(table)
        table = bigquery.Table(table_ref, schema=schema)
        table = bq_client.create_table(table)

    def execute_query(self, query, UseLegacySql=False):
        bq_client = self.client(self.project_id)
        job_config = bigquery.QueryJobConfig()
        job_config.use_legacy_sql = UseLegacySql
        query_job = bq_client.query(query, job_config=job_config)
        return query_job.result()

    def query2table(self, query, dest_table, dest_dataset, create_disposition=CREATE_IF_NEEDED,
                    write_disposition=WRITE_APPEND, UseLegacySql=False):
        bq_client = self.client(self.project_id)
        job_config = bigquery.QueryJobConfig()
        job_config.use_legacy_sql = UseLegacySql
        job_config.create_disposition = create_disposition
        job_config.write_disposition = write_disposition
        table_ref = bq_client.dataset(dest_dataset).table(dest_table)
        job_config.destination = table_ref
        query_job = bq_client.query(query, job_config=job_config)
        return query_job.result()

    def load(self, dataset, table, path, source_format, schema=None, bad_record_limit=0,
             skip_leading_rows=0, write_flag=WRITE_APPEND, ad_flag=False, allow_jagged_rows=False,
             create_disposition=CREATE_IF_NEEDED, ignore_unknown_values=False,
             allow_quoted_newlines=False, clustering_fields=None):
        bq_client = self.client(self.project_id)
        dataset_ref = self.bq_client.dataset(dataset)
        job_config = bigquery.LoadJobConfig()
        job_config.max_bad_records = bad_record_limit
        if schema:
            job_config.schema = schema
        # source_format - CSV/JSON
        job_config.source_format = source_format
        job_config.autodetect = ad_flag
        job_config.allow_jagged_rows = allow_jagged_rows
        job_config.allow_quoted_newlines = allow_quoted_newlines
        job_config.clustering_fields = clustering_fields
        # write disposition options: WRITE_TRUNCATE, WRITE_APPEND, WRITE_EMPTY
        job_config.write_disposition = write_flag
        job_config.create_disposition = create_disposition
        job_config.ignore_unknown_values = ignore_unknown_values
        if source_format == CSV:
            job_config.skip_leading_rows = skip_leading_rows
        load_job = bq_client.load_table_from_uri(path, dataset_ref.table(table),
                                                 job_config=job_config)

        return load_job.result()

    def bqtable2gs(self, dataset_name, table_name, bucket, gspath, file_name,
                   file_format=CSV, compression=False, ext='.gzip'):
        bq_client = self.client(self.project_id)
        job_config = bigquery.ExtractJobConfig()
        job_config.destination_format = file_format
        dataset_ref = bq_client.dataset(dataset_name, project=self.bq_project_id)
        if compression:
            job_config.compression = 'GZIP'
            file_name = file_name + ext
        destination_uri = 'gs://' + path.join(bucket, gspath, file_name)
        extract_job = \
            bq_client.extract_table(dataset_ref.table(table_name), destination_uri,
                                    job_config=job_config)
        return extract_job.result(), destination_uri

    def list_blobs(self, bucket_name, path, endsWith=''):
        objects = []
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)

        blobs = bucket.list_blobs(prefix=path)
        for blob in blobs:
            if blob.name.endswith(endsWith):
                objects.append(blob.name)
        return objects
