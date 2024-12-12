from xmlrpc.client import boolean
import pandas as pd
import numpy as np
from os import path
import logging
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_string_dtype
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

from core import am_sys_definitions
from core.gcp import Bigquery
from system import toolbox


def gen_mlb_col_names(key, classes: list):
    """
    Generale Multilabel Binalizer column names. Expects list of class names
    classes: list of classes
    """
    return ["{k}_{s}".format(k=key, s=s) for s in classes]


def get_model_configuration(model_name: str, submodels=[], work_dir="/tmp"):
    """
    Reads model configuration from from OS
    model_name: Model mame
    submodels: List of submodels to load
    """
    logger = logging.getLogger(__name__)
    model_config_gobal = toolbox.open_json_file(
        path.join(work_dir, model_name, am_sys_definitions.MODEL_CONFIG_GLOBAL_FILE)
    )
    all_submodels = toolbox.open_json_file(
        path.join(work_dir, model_name, am_sys_definitions.SUBMODEL_PARAMS_FILE)
    )
    if len(submodels) == 0:
        submodels_to_run = all_submodels
    else:
        submodels_to_run = {k: v for k, v in all_submodels.items() if k in submodels}

    logger.info("Global configuration is: \n {}".format(model_config_gobal))

    return model_config_gobal, submodels_to_run


def gen_pca_columns(pc_count: int):
    """
    Generate PCA columns list.
    pc_count: principal components count
    """
    return ["pc_{}".format(i) for i in range(pc_count)]


class RawData(object):
    """
    The class defines the initial datasets
    Arguments:
    df: pandas dataframe, containing the entire dataset (both independent and predicted if available)
    key_list: List of columns that form a unique key
    label_column: column name that acts as a label
    continious_label: boolean, if True, the label will be treated as scalar, otherwise as a discrete value
    """

    def __init__(
        self,
        df: pd.DataFrame,
        key_list: list,
        continious_label=None,
        label_column=None,
    ):
        """
        df: Raw data dataframe
        key_list: list of columns that constitute primary key
        continious label: specify if the column is continious
        label_column: name of the label column. Null if the mmodel is unsupervised
        """
        self.logger = logging.getLogger(__name__)

        # check if dataframe is not empty
        if df.empty:
            raise ValueError("The Dataframe passed is empty.")

        self.key_list = key_list
        drop_columns = key_list.copy()
        if label_column:
            drop_columns.append(label_column)
        self.logger.info("Drop columns are: {}".format(drop_columns))
        self.pred = df.drop(drop_columns, axis=1)
        self.logger.info("Prediction raw dataset shape is {}".format(self.pred.shape))
        if len(self.key_list) > 0:
            self.keys = df[key_list]
        else:
            self.keys = pd.DataFrame(df.index.values, index=df.index.values)
        self.y_labels, self.Y = self.treat_label_column(
            df, continious_label, label_column
        )

    def treat_label_column(self, df: pd.DataFrame, continious_label, label_column):
        """
        Thr function treats the label column, converts it into array if needed
        """
        if label_column in df.columns.values:
            label = df[label_column]
            if continious_label:
                return label_column, label
            else:
                dummies = pd.get_dummies(label)
                self.logger.info("classes distribution: {}".format(dummies.sum(axis=0)))
                return list(dummies.columns), dummies.values
        else:
            return None, []

    def _apply_pca(self, df, pca_model):
        data = pca_model.transform(df.values)

        return pd.DataFrame(data, columns=gen_pca_columns(data.shape[1]))

    def transform_x(self, prod_columns_definition: dict, prod_columns, pca_model=None):
        """
        Transforms independent set of into a model ready structure
        prod_column_definition: dictionary of columns names and their properties
        """
        self.logger.info("Raw data: \n {}".format(self.pred.head()))
        df_x_transform = pd.DataFrame(index=self.keys.index.values)
        for source_col_name, column_definition in prod_columns_definition.items():
            self.logger.info("Reconstructing column {}".format(source_col_name))
            if column_definition.get("type") == am_sys_definitions.REGULAR:
                scaler = column_definition.get("scaler")
                scaled_array = scaler.transform(self.pred[[source_col_name]])
                df_x_transform[source_col_name] = [i[0] for i in scaled_array]
            elif column_definition.get("type") == am_sys_definitions.CATEGORIAL:
                class_values = column_definition.get("values")
                if is_numeric_dtype(self.pred[source_col_name]):
                    self.pred[source_col_name] = self.pred[source_col_name].astype(int)
                    class_values = [int(i) for i in class_values]
                dummies = pd.get_dummies(
                    self.pred[[source_col_name]].astype(str)
                ).fillna(0)
                if sorted(class_values) == [0, 1]:
                    for d_col in dummies.columns.values:
                        if d_col == "{}_0".format(source_col_name):
                            dummies.drop([d_col], axis=1, inplace=True)
                df_x_transform = df_x_transform.merge(
                    dummies, how="inner", left_index=True, right_index=True
                )
            elif column_definition.get("type") == am_sys_definitions.MULTILABEL:
                mlb = column_definition.get("mlb")
                existingCols = gen_mlb_col_names(
                    key=source_col_name, classes=mlb.classes_
                )
                # create multilabel binarizer for existing data
                mlbDf = pd.DataFrame(
                    mlb.transform(self.pred[source_col_name].str.split(",")),
                    columns=existingCols,
                    index=self.keys.index.values,
                )
                df_x_transform = df_x_transform.merge(
                    mlbDf, how="inner", left_index=True, right_index=True
                )
            else:
                raise Exception("Unknown data type")
        self.logger.info("Reconstructed DF shape is {}".format(df_x_transform.shape))
        if pca_model:
            return self._apply_pca(df_x_transform, pca_model)
        else:
            for e in prod_columns:
                if e not in df_x_transform.columns.values:
                    df_x_transform[e] = 0
            transformed_data = df_x_transform[prod_columns]
            self.logger.info(
                "Reconstructed and transformed DF shape is {}".format(
                    transformed_data.shape
                )
            )
            return transformed_data


class ProcessedData(RawData):
    """
    The class inherits RawData class and transforms the data into model-ready structure
    Arguments:
    max_cat_val: int, maximal number of unique categories in categorial column, default 100
    multilabel_split: str, multilabel column value split, default `,`
    max_cat_ratio: float, the ratio between unique and regular count of cases in a column, below which the column is considered categorial.
                   Works with max_cat_val
    scale_regular: boolean, whether to scale the regular columns, default = True
    """

    def __init__(
        self,
        df: pd.DataFrame,
        key_list=[],
        label_column="y",
        max_cat_val=100,
        multilabel_split=",",
        max_cat_ratio=0.0001,
        scale_regular=True,
        balance_ratio=0,
        continious_label=True,
        low_info_thresh=0.99,
        primary_components=0,
        purpose=None,
    ):
        super().__init__(df, key_list, continious_label, label_column)
        self.max_cat_val = max_cat_val
        self.multilabel_split = multilabel_split
        self.low_info_thresh = low_info_thresh
        self.max_cat_ratio = max_cat_ratio
        self.scale_regular = scale_regular
        self.balance_ratio = balance_ratio
        self.primary_components = primary_components
        self.prod_columns = list(df.columns.values)
        self.pca_model = None
        self.pi = None
        self.mu = None
        self.purpose = purpose
        self.logger.info("Purpose of the model is {}".format(self.purpose))
        self.X, self.column_dict = self._treat_pred_columns()
        if not continious_label and self.balance_ratio > 0:
            self.pi, self.mu = self._undersample_data(ratio=self.balance_ratio)
            self.logger.info(
                "Shape of the prediction data is: {}".format(self.pred.shape)
            )
            self.logger.info("Shape of the label data is: {}".format(self.Y.shape))

    def _treat_pred_columns(self):
        df_x = pd.DataFrame(index=self.keys.index.values)
        column_dict = dict()
        for c in self.pred.columns.values.tolist():
            if self.purpose == "embedding":
                classification_dict = {}
            else:
                classification_dict = self._classify_column(self.pred[c])
                msg = "The column {col} is {tp}".format(
                    col=c, tp=classification_dict.get("type")
                )
                self.logger.info(msg)
            if classification_dict.get("type") == am_sys_definitions.REGULAR:
                scaler = preprocessing.StandardScaler().fit(self.pred[[c]])
                scaled_array = scaler.transform(self.pred[[c]])
                df_x[c] = [i[0] for i in scaled_array]
                classification_dict["scaler"] = scaler

            if self.purpose == "embedding":
                self.logger.info("Encoding column {}".format(c))
                encoder = preprocessing.LabelEncoder().fit(self.pred[c])
                encoded_array = encoder.transform(self.pred[c])
                df_x[c] = encoded_array
                classification_dict["encoder"] = encoder

            if classification_dict.get("type") == am_sys_definitions.CATEGORIAL:
                if is_numeric_dtype(self.pred[c]):
                    self.pred[c] = self.pred[c].astype(int)
                class_values = sorted(list(self.pred[c].unique()))
                classification_dict["values"] = [str(i) for i in class_values]
                dummies = pd.get_dummies(self.pred[[c]].astype(str)).fillna(0)
                for d_col in dummies.columns.values:
                    if self._drop_low_info(data=dummies[d_col]):
                        dummies.drop([d_col], axis=1, inplace=True)
                    if sorted(class_values) == [0, 1] and d_col == "{}_0".format(c):
                        self.logger.info("Dropping colunn {}".format(d_col))
                        dummies.drop([d_col], axis=1, inplace=True)
                df_x = df_x.merge(
                    dummies, how="inner", left_index=True, right_index=True
                )
            if classification_dict.get("type") == am_sys_definitions.MULTILABEL:
                mlb = preprocessing.MultiLabelBinarizer()
                mlb.fit(self.pred[c].str.split(self.multilabel_split))
                mlb_columns = gen_mlb_col_names(key=c, classes=mlb.classes_)
                df_mlb = pd.DataFrame(
                    mlb.transform(self.pred[c].str.split(self.multilabel_split)),
                    columns=mlb_columns,
                    index=self.keys.index.values,
                )
                for m_col in df_mlb.columns.values:
                    if self._drop_low_info(data=df_mlb[m_col]):
                        df_mlb.drop([m_col], axis=1, inplace=True)
                df_x = df_x.merge(
                    df_mlb, how="inner", left_index=True, right_index=True
                )
                classification_dict["mlb"] = mlb
                classification_dict["values"] = list(mlb.classes_)
            column_dict[c] = classification_dict
        # Getting the final list of the transformed data
        self.prod_columns = list(df_x.columns.values)
        self.logger.info("Ended building classification dict")
        if self.primary_components > 0:
            try:
                self.logger.info(
                    "Primary components: {}".format(self.primary_components)
                )
                return self._apply_pca(df_x), column_dict
            except Exception as ex:
                self.logger.info(
                    "Could not check for primary components module: {}".format(ex)
                )
        else:
            return df_x, column_dict

    def _apply_pca(self, df):
        pca = PCA(n_components=self.primary_components)
        self.pca_model = pca.fit(df.values)
        return pd.DataFrame(
            self.pca_model.transform(df.values),
            columns=gen_pca_columns(self.primary_components),
        )

    def _detect_cat_col(self, data):
        ser = pd.Series(data).astype(str)
        uniqueCases = ser.nunique()
        return (
            1.0 * uniqueCases / ser.count() < self.max_cat_ratio
            and uniqueCases <= self.max_cat_val
        )

    # Drop low info columns; Threshold is frequency of the value with the highest cardinality
    def _drop_low_info(self, data):
        # Getting number of rows for maximal frequency
        a = data.value_counts().max()
        if a / data.shape[0] > self.low_info_thresh:
            return True

    def _classify_column(self, array) -> str:
        def __cat_output(array):
            return {
                "type": am_sys_definitions.CATEGORIAL,
                "values": list(array.astype(str).unique()),
            }

        if is_numeric_dtype(array):
            # check if categorial
            if self._detect_cat_col(data=array.values):
                # if true then the column is categorial
                return __cat_output(array)
            else:
                return {"type": am_sys_definitions.REGULAR}
        else:
            try:
                if array.str.contains(self.multilabel_split).any():
                    return {"type": am_sys_definitions.MULTILABEL}
            except Exception as ex:
                self.logger.error(
                    "Could not transform array of type {} to string".format(array.dtype)
                )
            else:
                return __cat_output(array)

    def _undersample_data(self, ratio=1):
        """
        X: numpy 2D array, independent data
        y: numpy 1D array, labeled data
        ratio is the maximal ratio between the smallest category size and any other
        """
        # Distirbution of labels
        y_frequencies = np.sum(self.Y, axis=0)
        # finding minimal label with minimal frequency and its value
        min_frequency = int(min(y_frequencies))
        pi = int(y_frequencies[1]) / sum(y_frequencies)

        # indeces of records having minimal label
        list_x, list_y, list_ix, list_pred = list(), list(), list(), list()
        for i, _ in np.ndenumerate(y_frequencies):
            # getting label indeces
            ix = np.where(np.isin(self.Y[:, i[0]], 1))[0]
            # taking sample of minimal label frequency for each label
            smpl_ix = np.random.choice(
                a=ix, size=min(min_frequency * ratio, ix.shape[0]), replace=False
            )
            list_x.append(self.X.iloc[smpl_ix])
            list_pred.append(self.pred.iloc[smpl_ix])
            list_y.append(self.Y[smpl_ix])
            list_ix.append(smpl_ix)
        self.X = pd.concat(list_x)
        self.pred = pd.concat(list_pred)
        self.Y = np.concatenate(list_y)
        self.keys = self.keys.iloc[np.concatenate(list_ix)]
        y_frequencies = np.sum(self.Y, axis=0)

        min_frequency = int(min(y_frequencies))
        mu = min_frequency / sum(y_frequencies)

        return pi, mu


class Crossvalidation(KFold):
    """
    Object containing crossvalidation folds dictionary
    x: pandas datafram, independent dataset
    y: numpy array, labeled dataset
    folds: int, number of folds used in crossvalidation, Default 5

    Returns
    dictionary containing dictionaries with train and test folds
    """

    def __init__(self, x, y, folds=5) -> None:
        super().__init__(folds)
        self.folds = folds
        self.fold_dict = self._get_kfolds(folds=folds, x=x, y=y)

    def _get_kfolds(self, x, y):
        fold = 1
        kfold_dict = dict()
        for train_index, test_index in self.split(x):
            kfold_dict[fold] = {
                "x_train": x.iloc[list(train_index)],
                "y_train": y[train_index],
                "x_val": x.iloc[list(test_index)],
                "y_val": y[test_index],
            }
            fold += 1
        return kfold_dict


class TrainTestSplit(object):
    """
    Object containing train and test sets
    x: pandas datafram, independent dataset
    y: numpy array, labeled dataset
    train_pct: float, percent of data to keep for training

    Returns
    dictionary containing dictionaries with train and test folds
    """

    def __init__(self, x, y, train_pct=0.8) -> None:
        x.reset_index(drop=True, inplace=True)
        self.train_index = list(x.sample(frac=train_pct).index.values)
        self.test_index = x.loc[~x.index.isin(self.train_index)].index.values
        self.train_x = x.loc[self.train_index]
        self.train_y = y[self.train_index]
        self.test_x = x.loc[self.test_index]
        self.test_y = y[self.test_index]


class LocalDirectories(object):
    def __init__(self, model_name: str, bq=Bigquery) -> None:
        super().__init__()
        self.model_name = model_name
        self.logger = logging.getLogger(__name__)
        self.bq = bq

    def copy_from_gs(self, gs_bucket_name, gs_folder_name, work_dir="/tmp"):
        toolbox.deleteLocalFolder(path.join(work_dir, self.model_name))
        self.logger.info(
            "Copying model folder blobs to local folder from {}".format(
                path.join(gs_bucket_name, gs_folder_name, self.model_name)
            )
        )
        self.bq.copy_gs_folder_to_local(
            bucket_name=gs_bucket_name,
            source_path=path.join(gs_folder_name, self.model_name),
            dest_path=work_dir,
        )
        self.logger.info("Finished copying model definition")
