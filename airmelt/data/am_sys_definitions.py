QUERIES_FOLDER = "queries"
PROD_QUERIES_FOLDER = "prod_queries"
COLUMNS_FOLDER = "columns"
MODEL_RESULTS_FOLDER = "model_results"
PRODUCTION_MODELS_FOLDER = "production_models"
TEMPORARY_MODELS_FOLDER = "temporary_models"
MODEL_CONFIG_GLOBAL_FILE = "model_config_global.json"
SUBMODEL_PARAMS_FILE = "submodel_params.json"
COLUMNS_FILE = "columns.json"
EXPANDED_COLUMNS_FILE = "expanded_columns.txt"
STATISTICS_FILE = "statistics.json"
MODEL_OBJECT_NAME = "binary"
REGULAR = "regular"
MODEL_PICKLE_FILE = "model.pckl"
MODEL_JSON_FILE = "model.json"
PCA_PICKLE_FILE = "pca_model.pckl"
CATEGORIAL = "categorial"
MULTILABEL = "multilabel"
CATEGORIAL_OUTPUT_SCHEMA = [
    {"name": "ts", "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "target_column_current_value", "type": "FLOAT", "mode": "NULLABLE"},
    {
        "name": "model_value",
        "type": "RECORD",
        "mode": "REPEATED",
        "fields": [
            {"name": "value", "type": "FLOAT", "mode": "NULLABLE"},
            {"name": "class", "type": "INTEGER", "mode": "NULLABLE"},
        ],
    },
    {"name": "model_name", "type": "STRING", "mode": "NULLABLE"},
    {"name": "submodel_name", "type": "STRING", "mode": "NULLABLE"},
    {"name": "id", "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "submodel_rank", "type": "INTEGER", "mode": "NULLABLE"},
]
CONT_OUTPUT_SCHEMA = [
    {"name": "ts", "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "target_column_current_value", "type": "FLOAT", "mode": "NULLABLE"},
    {"name": "model_value", "type": "FLOAT", "mode": "NULLABLE"},
    {"name": "model_name", "type": "STRING", "mode": "NULLABLE"},
    {"name": "submodel_name", "type": "STRING", "mode": "NULLABLE"},
    {"name": "id", "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "submodel_rank", "type": "INTEGER", "mode": "NULLABLE"},
]
GENERAL_OUTPUT_SCHEMA = [
    {"name": "ts", "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "target_column_current_value", "type": "FLOAT", "mode": "NULLABLE"},
    {
        "name": "model_value",
        "type": "RECORD",
        "mode": "REPEATED",
        "fields": [
            {"name": "measure", "type": "STRING", "mode": "NULLABLE"},
            {"name": "value", "type": "FLOAT", "mode": "NULLABLE"},
        ],
    },
    {"name": "model_name", "type": "STRING", "mode": "NULLABLE"},
    {"name": "submodel_name", "type": "STRING", "mode": "NULLABLE"},
    {"name": "id", "type": "INTEGER", "mode": "NULLABLE"},
    {"name": "submodel_rank", "type": "INTEGER", "mode": "NULLABLE"},
]
