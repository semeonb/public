from os import path

import boto3
import logging

from airmelt_data import gcp
from airmelt_system import toolbox


class SQS(object):
    """
    The class manages the AWS SQS queues

    """

    def __init__(self, aws_access_key_id, aws_secret_access_key, region_name: str):
        """
        name: Name of the queue
        aws_access_key_id: AWS access key
        aws_secret_access_key: AWS secret access key

        """
        self.logger = logging.getLogger(__name__)
        try:
            self.sqs_client = boto3.client(
                "sqs",
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            self.logger.info("Created sqs client")
        except Exception as ex:
            self.logger.error(ex)
            raise

    def send_message(self, queue_url, message_body, delay_seconds=0):

        response = self.sqs_client.send_message(
            QueueUrl=queue_url,
            DelaySeconds=delay_seconds,
            MessageBody=message_body,
        )
        return response

    def send_message_list(self, data, queue_url):
        sent = 0
        failed = 0
        received = 0
        for row in data:
            try:
                response = self.send_message(queue_url=queue_url, message_body=row)
                sent += 1
                sqs_message_id = response.get("MessageId")
                if sqs_message_id:
                    received += 1
                else:
                    self.logger.error(response)
                    failed += 1
            except Exception:
                failed += 1
        self.logger.info("rows sent: {sent}".format(sent=sent))
        if failed > 0:
            raise Exception("not all rows were sent")
        return {"sent": sent, "received": received, "failed": failed}

    def send_batch_list(self, data, queue_url, batch_size=10):
        failed = 0
        sent = 0
        received = 0
        for x in toolbox.batch(data, batch_size):

            entries = [
                {
                    "Id": str(ind),
                    "MessageBody": str(msg),
                }
                for ind, msg in enumerate(x)
            ]
            batch_failed = 0
            batch_sent = len(entries)
            response = self.sqs_client.send_message_batch(
                QueueUrl=queue_url, Entries=entries
            )
            if "Failed" in response:
                for msg_meta in response["Failed"]:
                    batch_failed += 1
                    self.logger.warning(
                        "Failed to send: %s: %s",
                        msg_meta["MessageId"],
                        x[int(msg_meta["Id"])]["MessageBody"],
                    )
            batch_received = batch_sent - batch_failed
            sent += batch_sent
            received += batch_received
            failed += batch_failed
        return {"sent": sent, "received": received, "failed": failed}

    def get_queue_url(self, queue_name):
        queue = self.sqs_client.get_queue_url(QueueName=queue_name)
        return queue["QueueUrl"]

    def transfer_bq_query(
        self, project_id, query, queue_name=None, queue_url=None, batch_size=1
    ):
        """
        Transfers the contnets of a table to SQS queue

        project_id: str: BQ project ID
        dataset_name: str: BQ dataset name
        table_name: str: BQ table name
        queue_name: str: Amazon SQS queue name
        queue_url: str: Amazon SQS queue url
        """
        sent = 0
        failed = 0
        received = 0
        bq = gcp.Bigquery(bq_project_id=project_id)
        # Get query results
        results = bq.query_to_list(query)

        # Get queue
        if queue_name:
            queue_url = self.get_queue_url(queue_name)
        else:
            if not queue_url:
                raise Exception("No queue defined")
        if batch_size == 1:
            output = self.send_message_list(results, queue_url)
        else:
            output = self.send_batch_list(results, queue_url)
        return output


class S3(object):
    """
    description: The class manages the AWS S3 buckets
    """

    def __init__(self, aws_access_key_id, aws_secret_access_key, region_name: str):
        """
        name: Name of the queue
        aws_access_key_id: AWS access key
        aws_secret_access_key: AWS secret access key

        """
        self.logger = logging.getLogger(__name__)
        try:
            self.s3_client = boto3.client(
                "s3",
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            self.logger.info("Created s3 client")
        except Exception as ex:
            self.logger.error(ex)
            raise

        # Get S3 session
        try:
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
            )
            self.s3_resource = session.resource("s3")
            self.logger.info("Created s3 resource")
        except Exception as ex:
            self.logger.error(ex)
            raise

    def upload_file(self, filename, bucket, folder):
        bucket = self.s3_resource.Bucket(bucket)
        destKey = path.join(folder, path.basename(filename))
        with open(filename, "rb") as data:
            bucket.put_object(Key=destKey, Body=data)
