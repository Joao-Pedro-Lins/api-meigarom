import pandas as pd
import sklearn
import numpy
from flask import Flask, request
import pickle

from wine_quality import WineQuality


## carrgando o modelo
model = pickle.load(open("modelo.pkl", "rb"))

## instanciando o flask
app = Flask(__name__)

## criando meus endpoints
@app.route('/predict', methods=['POST'])
def predict():
    ## coletando os dados
    test_json = request.get_json()
    if test_json:
        if isinstance(test_json, dict): #valor único
            df = pd.DataFrame(test_json, index=[0])
        else:
            df = pd.DataFrame(test_json)

    ## pre processing
    pipeline = WineQuality()
    df1 = pipeline.data_preparation(df)

    ## prediction
    pred = model.predict(df1)
    df1['prediction'] = pred
    response = df1

    return response.to_json(orient='records')

if __name__ == "__main__":
    ## start flask
    app.run(host="0.0.0.0", port="8888")
