import os
from functools import reduce
from datetime import datetime


import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


def getSparseCases(ser, sparseThresh=0.001, topCases=100, coverage=0.95):
    lst0 = ser.index.values.tolist()
    t_cases = ser.sum()
    
    lst1 = ser.nlargest(topCases).index.values.tolist()
    lst2 = ser.loc[ser/t_cases > sparseThresh].index.values.tolist()
    normalCases = [i for i in lst0 if i in lst1 and i in lst2]
    return [i for i in lst0 if i not in normalCases]


def getSparseCasesCov(ser, coverage=0.95):  
    t_cases = ser.sum()
    ser = ser/t_cases
    ser = ser.sort_values().cumsum()
    return ser.loc[ser < 1 - coverage].index.values.tolist()


def concatDf(colList):
    def obj(row):
        return '_'.join(row[colList])
    return obj


def dropLowInfo(df, thresh=0.0001, cat_prefix='cat_'):
    totalRows = df.shape[0]
    columnsToDrop = []
    for c in df.columns.values:
        if c.startswith(cat_prefix):
            # Getting number of rows for maximal frequency
            colValueCounts = df[c].value_counts().max()
            if 1 - colValueCounts/totalRows < thresh:
                columnsToDrop.append(c)
    return columnsToDrop
    

        
def index_marks(nrows, chunk_size):
    return range(1 * chunk_size, (nrows // chunk_size + 1) * chunk_size, chunk_size)


def split(dfm, chunk_size):
    indices = index_marks(dfm.shape[0], chunk_size)
    return np.split(dfm, indices)


def buildPivotColumn(orgCol, colVal):
    return '{pc}_{rep}'.format(pc=orgCol, rep=colVal.replace(' ', '_'))


def ordered_join(data):
    return reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True, how='inner'),
                  data)
    

def daysSince(columnName, lastDate, dateFormat):
    def obj(row):
        # difference in days
        return (lastDate - datetime.strptime(row[columnName], dateFormat)).days
    return obj


def daysSinceDf(df, columnName, lastDate, dateFormat='%Y-%m-%d %H:%M:%S'):
    newColumn = 'daysSince_' + columnName
    df[newColumn] = df.apply(daysSince(columnName, lastDate, dateFormat), axis=1)
    return df[[newColumn]]


def buildCategoricalDf(df, catThreshold, excludedColumns, maxCategories=100):
    dfList = []
    nonCatColumns = []
    dfIndex = df.index
    for col in df.columns:
        data = df[col].values
        if detectCategoricalColumns(data, catThreshold) and col not in excludedColumns:
            newCol = 'cat_{col}'.format(col=col)
            newDf = pd.DataFrame(data, columns=[newCol], index=dfIndex)
            newDf[newCol] = newDf[newCol].astype(str)
            dfList.append(newDf)
        else:
            nonCatColumns.append(col)    
    dfList.append(df[nonCatColumns])
    return ordered_join(dfList), nonCatColumns


def detectCategoricalColumns(data, threshold, maxCategories=100):
    ser = pd.Series(data).astype(str)
    uniqueCases = ser.nunique()
    return 1.*uniqueCases/ser.count() < threshold and uniqueCases < maxCategories


def generateCategories(columns, X, categorical_prefix='cat_'):
    relevant_vars = []
    dictionary = {}
    for i, c in enumerate(columns):
        if c.startswith(categorical_prefix):
            relevant_vars.append((c, i))
    for v in relevant_vars:
        tmp_cat = [item[v[1]] for item in X]
        dictionary[v[0]] = set(tmp_cat)
    return dictionary


def data_scaler(X):
    scaler = preprocessing.StandardScaler().fit(X)
    return scaler.transform(X), scaler


def createScaledArray(X, dictionary, expPredictors=None, mlbTransDict={}, scaler=None,
                      keyColumnName=None, train=True):
    regular_columns = []
    encoded_columns = []
    multilabel_columns = []

    def genColNames(key, classes):
        return [key + '_' + s for s in classes]

    for i, v in enumerate(X.columns.values.tolist()):
        df = X[[v]]
        if v in dictionary:
            categories = dictionary[v]
            df = generateDummyVars(df=df, categories=categories)
            df.columns = [v + '_' + str(i) for i in df.columns.values]
            encoded_columns.append(df)
        elif v in mlbTransDict:
            mlbTrans = mlbTransDict[v]
            existingCols = genColNames(v, mlbTrans.classes_)
            # create multilabel binarizer for existing data
            if train:
                df = pd.DataFrame(mlbTrans.transform(X[v].str.split(',')),
                                  columns=existingCols, index=X[v].index)
            else:
                mlb = preprocessing.MultiLabelBinarizer()
                df = pd.DataFrame(mlb.fit_transform(X[v].str.split(',')),
                                  columns=genColNames(v, mlb.classes_), index=X[v].index)
                for c in existingCols:
                    if c not in df.columns.values.tolist():
                        df[c] = 0
                for c in df.columns.values.tolist():
                    if c not in existingCols:
                        df.drop(c, axis=1, inplace=True)
            multilabel_columns.append(df)
            if set(df.columns.values.tolist()) != set(existingCols):
                print('Multilabel columns are not as previous ones!')
                raise Exception('Multilabel columns are not as previous ones!')
            else:
                print('Multilabel columns are identical')
        else:
            regular_columns.append(pd.DataFrame(data_scaler(df)[0], columns=[v], index=df.index))
        X.drop(columns=[v], axis=1)
    finalDfList = []
    # merge dataframes in encoded columns
    if len(encoded_columns) > 0:
        dfEncoded = ordered_join(encoded_columns)
        finalDfList.append(dfEncoded)
    # merge regular columns
    if len(regular_columns) > 0:
        finalDfList.append(ordered_join(regular_columns))
    # merge list variables columns
    if len(multilabel_columns) > 0:
        dfMultilabel = ordered_join(multilabel_columns)
        finalDfList.append(dfMultilabel)

    data = ordered_join(finalDfList)
    del(finalDfList)
    if expPredictors:
        return data[expPredictors]
    else:
        return data
    
def generateDummyVars(df, categories):
    dummies = pd.get_dummies(df, prefix='', prefix_sep='')
    return dummies.T.reindex(categories).T.fillna(0)


def color_map(group_vals, cmap_code):

    # Set the color map to match the number of groups
    cmap = plt.get_cmap(cmap_code)
    cNorm = colors.Normalize(vmin=0, vmax=len(group_vals))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    return scalarMap


def group_vals_funct(df, category_col, cmap_code='Accent'):
    DEFAULT_CATEGORY_COLUMN_NAME = 'default_category_column'
    DEFAULT_CATEGORY_COLUMN_VALUE = 'Total'
    if (category_col is None):
        category_col = DEFAULT_CATEGORY_COLUMN_NAME
        df[category_col] = DEFAULT_CATEGORY_COLUMN_VALUE
    group_vals = df[category_col].unique()
    return df, group_vals, color_map(group_vals, cmap_code=cmap_code), category_col


def precisionRecall(pred, y):
    df = pd.DataFrame(np.stack([pred, y], axis=1), columns=['pred', 'y'])
    precision_recall_plot(df=df, figsize=(8, 8), score_col_name='pred', test_col_name='y',
                          show_legend=True, fontsize=12)
    

def precision_recall_plot(df, score_col_name, test_col_name, category_col=None,
                          show_legend=True, alpha=1, markersize=1, figsize=(10, 10), fontsize=12,
                          where='post'):
    """
    precision_recall_plot
    Author: Semeon Balagula
    Functionality: receives pandas dataframe. Plots precision recall plot for different models.
    The chart to be plotted is step chart. Average precision will be plotted in the legend
    score_col_name - the name of the column to use as model score
    test_col_name - the name of the column to regard as truth
    category_col - the name of the category column. Plot will paint different models in
                   different color
    alpha - Set the alpha value used for blending. Values are between 0 and 1
    figsize - size of the plot to display
    fontsize - font size of labels and legends
    show_legend (Boolean) - indicates if need to show legend
    where - {'pre', 'post', 'mid'}, optional, default 'post'
            Define where the steps should be placed
    """
    df, group_vals, scalar_map, category_col = group_vals_funct(df=df, category_col=category_col)
    plt.figure(figsize=figsize)

    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')

    for color_idx, val in enumerate(group_vals):
        data_group = df[(df[category_col] == val)]
        y_test = np.array(data_group[test_col_name])
        y_score = np.array(data_group[score_col_name])
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        average_precision = average_precision_score(y_test, y_score)
        plt.step(recall, precision, color=scalar_map.to_rgba(color_idx),
                 alpha=alpha, where=where,
                 label=val + ' :Integral = {0:0.2f}'.format(average_precision))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    if show_legend:
        plt.legend(loc='best')
        plt.legend(fontsize=fontsize)
    plt.show()
    

def plot_confusion_matrix(pred, y, names, title='Confusion matrix', cmap=plt.cm.Blues,
                          normalize=False):
    pred = np.argmax(pred, axis=1)
    y = np.argmax(y, axis=1)
    cm = confusion_matrix(y, pred)
    np.set_printoptions(precision=2)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=names, yticklabels=names,
           title=title, ylabel='True label', xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right',
             rotation_mode='anchor')
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    plt.show()