import setuptools


setuptools.setup(
    name="airmelt_utils",
    version="0.1.3",
    author="Semeon Balagula @ Airmelt",
    description="Set of utilities",
    packages=["airmelt_data", "airmelt_system"],
    install_requires=[
        "importlib-metadata==4.8.3",
        "cryptography==41.0.3",
        "scikit-learn==0.24.0",
        "Keras==2.4.3",
        "tensorflow==2.3.1",
        "joblib==0.13.2",
        "Keras-Preprocessing==1.1.2",
        "numpy==1.18.5",
        "pandas==0.24.2",
        "google-cloud-bigquery==2.20.0",
        "google-cloud-storage==1.36.1",
        "google-cloud-core==1.7.0",
        "google-cloud-bigquery-storage==2.4.0",
        "google-cloud==0.34.0",
        "docopt==0.6.2",
        "slacker==0.13.0",
        "Lifetimes==0.11.3",
        "pyarrow==4.0.1",
        "azure-core==1.26.1",
        "azure-identity==1.12.0",
        "azure-storage-blob==12.14.1",
        "sqlparse==0.4.3",
        "boto3==1.17.101",
        "pyodbc==4.0.35",
        "mysql-connector-python==8.0.31",
        "markupsafe==2.0.0",
        "xgboost==1.6.2",
    ],
)
