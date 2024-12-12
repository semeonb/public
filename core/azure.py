from azure.storage.blob import ContainerClient


class AzureContainerClient(object):
    def __init__(self, account_url: str):
        self.client = ContainerClient.from_container_url(container_url=account_url)

    def list_blobs_name(self):
        blobs = self.client.list_blobs()
        return [blob.get("name") for blob in blobs]

    def download_file(self, destination_path, source_path, mode="wb"):
        with open(file=destination_path, mode=mode) as download_file:
            download_file.write(self.client.download_blob(source_path).readall())
        return True
