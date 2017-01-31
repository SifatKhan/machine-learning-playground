from flask import Flask,send_file
import pandas as pd
import flask
app = Flask(__name__)

dataframe = pd.read_csv('train.csv')


@app.route("/")
def hello():
    return send_file("static/index.html")

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
        neigh['type'] = 'box'
        data.append(neigh)

    return flask.jsonify(data)



@app.route("/api/yearlyAvg")
def yearly_avg():
    data = {}
    data['x'] = dataframe.groupby('YrSold').SalePrice.mean().index.astype(str).tolist()
    data['y'] = dataframe.groupby('YrSold').SalePrice.mean().tolist()

    return flask.jsonify(data)


if __name__ == "__main__":
    app.run()


