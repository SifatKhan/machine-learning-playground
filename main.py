from flask import Flask, send_file
from scipy import stats
import pandas as pd
import numpy as np
import flask

app = Flask(__name__, static_folder='HousePrices/static/',static_url_path='')

dataframe = pd.read_csv('HousePrices/data/train.csv')


@app.route("/")
def hello():
    return send_file("HousePrices/static/index.html")


@app.route("/api/neighborhood/counts")
def neighborhood_counts():
    data = {}
    data['type'] = "bar"
    data['x'] = dataframe.Neighborhood.value_counts().index.tolist()
    data['y'] = dataframe.Neighborhood.value_counts().tolist()

    return flask.jsonify(data)


@app.route("/api/neighborhood/boxplot")
def neighborhood_boxplot():
    data = []
    neighborhoods = dataframe.Neighborhood.unique().tolist()
    for n in neighborhoods:
        neigh = {}
        neigh['name'] = n
        neigh['y'] = dataframe[dataframe['Neighborhood'] == n].SalePrice.tolist()
        neigh['median'] = dataframe[dataframe['Neighborhood'] == n].SalePrice.median()
        neigh['type'] = 'box'
        data.append(neigh)

    data = sorted(data, key=lambda m: m['median'],reverse=True)
    allpoints = {}
    allpoints['name'] = 'All'
    allpoints['y'] = dataframe.SalePrice.tolist()
    allpoints['type'] = 'box'
    data.append(allpoints)
    lol = 23

    return flask.jsonify(data)


@app.route("/api/pricing/grlivarea")
def pricing_grlivarea():
    data = {}
    data['type'] = 'scatter'
    data['mode'] = 'markers'
    df = dataframe[(dataframe['GrLivArea'].notnull())][['GrLivArea', 'SalePrice']]
    df = df[(np.abs(stats.zscore(df)) < 4).all(axis=1)] # Remove outliers beyond 4 sigmas
    data['y'] = df[(df['GrLivArea'].notnull())]['SalePrice'].values.tolist()
    data['x'] = df[(df['GrLivArea'].notnull())]['GrLivArea'].values.tolist()
    return flask.jsonify([data])


@app.route("/api/pricing/lotfrontage")
def pricing_lotfrontage():
    data = {}
    data['type'] = 'scatter'
    data['mode'] = 'markers'
    df = dataframe[(dataframe['LotFrontage'].notnull())][['LotFrontage', 'SalePrice']]
    df = df[(np.abs(stats.zscore(df)) < 4).all(axis=1)] # Remove outliers beyond 4 sigmas
    data['y'] = df[(df['LotFrontage'].notnull())]['SalePrice'].values.tolist()
    data['x'] = df[(df['LotFrontage'].notnull())]['LotFrontage'].values.tolist()
    return flask.jsonify([data])


@app.route("/api/pricing/lotarea")
def pricing_lotarea():
    data = {}
    data['type'] = 'scatter'
    data['mode'] = 'markers'

    df = dataframe[(dataframe['LotFrontage'].notnull())][['LotArea','SalePrice']]
    df = df[(np.abs(stats.zscore(df)) < 4).all(axis=1)] # Remove outliers beyond 4 sigmas

    data['y'] = df['SalePrice'].values.tolist()
    data['x'] = df['LotArea'].values.tolist()

    return flask.jsonify([data])


@app.route("/api/pricing/histogram")
def histogram():
    data = {}
    data['type'] = 'histogram'
    data['boxmean'] = True
    data['x'] = dataframe.SalePrice.values.tolist()

    response = {}
    response['data'] = [data]
    response['mean'] = dataframe.SalePrice.mean()

    return flask.jsonify(response)


@app.route("/api/correlation")
def correlation():
    data = {}
    data['type'] = 'heatmap'
    df = dataframe.drop('Id', axis=1)
    data['z'] = df.corr().as_matrix().tolist()
    data['x'] = df.corr().index.tolist()
    data['y'] = df.corr().index.tolist()

    return flask.jsonify([data])




@app.route("/api/yearly/avg")
def yearly_avg():
    data = {}
    df = dataframe
    df['YrSold'] = pd.to_datetime(dataframe.YrSold, format='%Y')
    data['x'] = df.groupby('YrSold').SalePrice.mean().index.astype(str).tolist()
    data['y'] = dataframe.groupby('YrSold').SalePrice.mean().tolist()

    return flask.jsonify(data)


@app.route("/api/yearly/count")
def yearly_count():
    data = {}
    df = dataframe
    df['YrSold'] = pd.to_datetime(dataframe.YrSold, format='%Y')

    data['x'] = df.groupby('YrSold').YrSold.count().index.astype(str).tolist()
    data['y'] = df.groupby('YrSold').YrSold.count().tolist()

    return flask.jsonify(data)


@app.route("/api/monthly/count")
def monthly_count():
    data = {}
    df = dataframe
    df['MoSold'] = pd.to_datetime(dataframe.MoSold, format='%m')

    data['x'] = df.groupby('MoSold').MoSold.count().index.astype(str).tolist()
    data['y'] = df.groupby('MoSold').MoSold.count().tolist()
    data['type'] = 'bar'

    return flask.jsonify(data)


if __name__ == "__main__":
    app.run()
