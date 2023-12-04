from zipfile import ZipFile

import pycountry
import numpy as np
import pandas as pd

import plotly.io as pio
import plotly.express as px

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    precision_recall_fscore_support,
    accuracy_score,
    RocCurveDisplay,
    roc_auc_score,
    roc_curve,
)

from .constants import (
    AGE_CATE,
    AGE_CAT,
    AGE_CAT_RANGE,
    GENDER_CAT,
    COLUMNS_MAJORITY,)

# Set plotly theme dark
pio.templates.default = 'plotly_dark'

# Function to convert country to any ISO country format
def country_conversion(country, code='numeric'):
    try:
        return getattr(pycountry.countries.lookup(country), code)
    except LookupError:
        # It is a missing data and cannot be convert 'NaN'
        # value of country to numeric code
        return -1 if code == 'numeric' else 'Unknown'
    except ArithmeticError as att:
        # In case of the code requested is wrong
        raise ArithmeticError(
            f'The City object from "pycountry" does not have Attribute {code}') from att


# Function to display interactively rules
# Filtered by min threshold for a given metric
def rules_display(
        data: pd.DataFrame,
        _value, _metric,
        _antecedents, _consequents):
    print(f'Filtering rules by "{_metric}" >= {_value}')
    _filtered = data.query(f'{_metric} >= @_value')\
        .sort_values(_metric)
    # TODO: improve antecedent and consequent filtering...
    # Enabling multi selection of items
    if _antecedents:
        print(f'Filtering rules with antecedent: "{_antecedents}"')
        _mask = data['antecedents'].astype('str').str.contains(Fr"(.*?){_antecedents}", case=False, regex=True)
        _filtered = _filtered[_mask]
    if _consequents:
        print(f'Filtering rules with consequent: "{_consequents}"')
        _mask = data['consequents'].astype('str').str.contains(Fr"(.*?){_consequents}", case=False, regex=True)
        _filtered = _filtered[_mask]
    return _filtered


# Function to display interactively filtered by length of set and support
def items_sets_display(
        data: pd.DataFrame,
        _length,
        _support):
    print(
        f'Filtering items sets of "length" {_length} and "Support" >= {_support}')
    return data.query('length == @_length and support >= @_support')\
        .sort_values('support', ascending=False)


# Function to read and clean integrity of data
def load_data(path, use_cols=True):
    data_ = None
    with ZipFile(path) as zip_file:
        for compressed in zip_file.filelist:
            if compressed.filename.endswith('csv'):
                with zip_file.open(compressed) as data_file:
                    data_ = pd.read_csv(
                        data_file,
                        sep=';',
                        header=0,
                        low_memory=False,
                        usecols=range(1, 64) if use_cols else None,)
    # Convert to categorical types
    if 'gender' in data_.columns:
        data_['gender'] = data_['gender'].astype(GENDER_CAT)
        data_['ageBroad'] = data_['ageBroad'].astype(AGE_CAT_RANGE)
        data_[COLUMNS_MAJORITY] = data_[COLUMNS_MAJORITY].astype(AGE_CAT)

    if 'Datasource' in data_.columns:
        data_.drop(columns=['Datasource'], inplace=True)
    return data_


# Fix majority columns integrity
def fix_integrity(data: pd.DataFrame):
    # Dropping Incongruent due to anonymization
    # Total of records to drop 12
    indexes = [
        # majorityStatus = Adult  majorityStatusAtExploit = Minor	majorityEntry = Adult
        data.query(
            'majorityStatus == @AGE_CATE[0] & majorityStatusAtExploit == @AGE_CATE[1] & majorityEntry == @AGE_CATE[0]').index,
        # majorityStatus = Minor  majorityStatusAtExploit = NaN	majorityEntry = Adult
        data.query(
            'majorityStatus == @AGE_CATE[1] & majorityStatusAtExploit.isna() & majorityEntry == @AGE_CATE[0]').index,
        # majorityStatus = NaN  majorityStatusAtExploit = Minor	majorityEntry = Adult
        data.query(
            'majorityStatus.isna() & majorityStatusAtExploit == @AGE_CATE[1] & majorityEntry == @AGE_CATE[0]').index,
    ]

    drop_index = pd.Index([])
    for index in indexes:
        drop_index = drop_index.join(index, how='outer')
    data.drop(drop_index, inplace=True)

    # Completing missing data due to anonymization
    # Total of records to fill 26778
    indexes = [
        # majorityStatus = Adult  majorityStatusAtExploit = Minor	majorityEntry = NaN
        (
            data.query(
                'majorityStatus == @AGE_CATE[0] & majorityStatusAtExploit == @AGE_CATE[1] & majorityEntry.isna()'
            ).index,
            'Minor',
            'majorityEntry'),
        # majorityStatus = Adult  majorityStatusAtExploit = NaN	majorityEntry = Adult
        (
            data.query(
                'majorityStatus == @AGE_CATE[0] & majorityStatusAtExploit.isna() & majorityEntry == @AGE_CATE[0]'
            ).index,
            'Adult',
            'majorityStatusAtExploit'),
        # majorityStatus = Minor  majorityStatusAtExploit = Minor	majorityEntry = NaN
        (
            data.query(
                'majorityStatus == @AGE_CATE[1] & majorityStatusAtExploit == @AGE_CATE[1] & majorityEntry.isna()'
            ).index,
            'Minor',
            'majorityEntry'),
        # majorityStatus = Minor  majorityStatusAtExploit = NaN	majorityEntry = Minor
        (
            data.query(
                'majorityStatus == @AGE_CATE[1] & majorityStatusAtExploit.isna() & majorityEntry == @AGE_CATE[1]'
            ).index,
            'Minor',
            'majorityStatusAtExploit'),
        # majorityStatus = Minor  majorityStatusAtExploit = NaN	majorityEntry = NaN
        (
            data.query(
                'majorityStatus == @AGE_CATE[1] & majorityStatusAtExploit.isna() & majorityEntry.isna()'
            ).index,
            'Minor',
            ['majorityStatusAtExploit', 'majorityEntry']),
        # majorityStatus = NaN  majorityStatusAtExploit = Adult	majorityEntry = Adult
        (
            data.query(
                'majorityStatus.isna() & majorityStatusAtExploit == @AGE_CATE[0] & majorityEntry == @AGE_CATE[0]'
            ).index,
            'Adult',
            'majorityStatus'),
        # majorityStatus = NaN  majorityStatusAtExploit = Adult	majorityEntry = NaN
        (
            data.query(
                'majorityStatus.isna() & majorityStatusAtExploit == @AGE_CATE[0] & majorityEntry.isna()'
            ).index,
            'Adult',
            ['majorityStatus', 'majorityEntry']),
        # majorityStatus = NaN  majorityStatusAtExploit = Minor	majorityEntry = Minor
        (
            data.query(
                'majorityStatus.isna() & majorityStatusAtExploit == @AGE_CATE[1] & majorityEntry == @AGE_CATE[1]'
            ).index,
            'Minor',
            'majorityStatus'),
        # majorityStatus = NaN  majorityStatusAtExploit = Minor	majorityEntry = NaN
        (
            data.query(
                'majorityStatus.isna() & majorityStatusAtExploit == @AGE_CATE[1] & majorityEntry.isna()'
            ).index,
            'Minor',
            ['majorityStatus', 'majorityEntry']),
        # majorityStatus = NaN  majorityStatusAtExploit = NaN	majorityEntry = Adult
        (
            data.query(
                'majorityStatus.isna() & majorityStatusAtExploit.isna() & majorityEntry == @AGE_CATE[0]'
            ).index,
            'Adult',
            ['majorityStatus', 'majorityEntry']),
    ]

    for index, value, columns in indexes:
        data.loc[index, columns] = value


# Function to generate plot of world country heatmap
def heatmap_country(data: pd.DataFrame, key, title):
    # Heat map of citizen ship of individual
    map_raw = data[key].dropna().apply(
        country_conversion, code='alpha_3')
    map_raw = map_raw.value_counts().reset_index(name='records')
    map_raw.sort_values('records', ascending=False)
    map_fig = px.choropleth(
        map_raw,
        locations=key,
        color='records',
        hover_name=map_raw[key].apply(
            country_conversion, code='name'),
        hover_data={
            key: False,
        },
        color_continuous_scale="burgyl",
        projection='natural earth',
        title=title,
        template='plotly_white'
    )

    map_fig.update_layout(
        height=600,
        width=1000,)

    map_fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br><br>Persons: %{z}<extra></extra>'
    )

    return map_fig


# Function to generate plot for PCA
def scatter_pca(data: pd.DataFrame, color, title, facet_col=None, add_hover=True):
    scatter_fig = px.scatter(
        data,
        x='x_0',
        y='x_1',
        opacity=.8,
        color=color,
        title=title,
        facet_col=facet_col,
        hover_name='ageBroad',
        marginal_x='histogram',
        hover_data={
            'gender': add_hover,
            'majorityStatus': add_hover
        }
    )

    if facet_col:
        scatter_fig.for_each_annotation(
            lambda annotation: annotation.update(text=annotation.text.replace('gender=', '')))
    scatter_fig.update_traces(
        selector=dict(type='scattergl'),
        hovertemplate='<b>Age Broad:</b> <i>%{hovertext}</i><br><br>x_0: %{x:.2f}<br>x_1: %{y:.2f}<extra></extra>'
    )

    scatter_fig.update_traces(
        selector=dict(type='histogram'),
        hovertemplate='<b>x_0:</b> %{x}<br><br>Count: %{y}<extra></extra>',
    )
    return scatter_fig


# Feature selection by variance threshold
def variance_columns(data, threshold):
    # TODO: Check which estimator could be use in a SequentialFeatureSelector
    return VarianceThreshold(threshold).fit(data).get_feature_names_out()


# Function to generate plot for Association rules
def scatter_rules(rules: pd.DataFrame, title):
    # the size and color of the marker is the lift score
    rules_plot = px.scatter(
        rules,
        x='support',
        y='confidence',
        size='lift',
        color='lift',
        opacity=.4,
        title=title,
        text=rules.index,
    )

    rules_plot.update_traces(
        hovertemplate='<b>Lift: %{marker.size:.3f}</b><br><br><i>Rule index: %{text}</i><br>Support: %{x:.3f}<br>Confidence: %{y:.3f}'
    )

    return rules_plot


# Function to retrieve the performance evaluation metrics for
# the predictions of a classification model
def eval_metrics(predict: np.ndarray, real: np.ndarray):
    if predict.shape[0] != real.shape[0]:
        raise ValueError(f'Prediction and real class must have same dim, instead got {predict.shape} {real.shape}')
    metrics = precision_recall_fscore_support(
        real,
        predict,
        average='macro',
        zero_division=np.nan)[:-1]
    return dict(
        zip(('f_score', 'precision', 'recall'), metrics), accuracy=accuracy_score(real, predict))


# Function to generate plot of ROC curve
def make_roc_curve(
        predict: np.ndarray,
        real: np.ndarray,
        pos_label: str,
        estimator_name):
    encoder = LabelEncoder().fit(real)
    label_position = encoder.transform([pos_label])[0]
    y_real = encoder.transform(real)
    fpr, tpr, _ = roc_curve(
        y_real,
        predict[:, label_position],
        pos_label=label_position)
    auc = roc_auc_score(y_real, predict[:, label_position], average='micro', multi_class='ovr')
    return RocCurveDisplay(
        fpr=fpr, tpr=tpr, roc_auc=auc,
        estimator_name=estimator_name)
