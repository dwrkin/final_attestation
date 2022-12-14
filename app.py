import flask
from flask import render_template
from tensorflow.keras.models import load_model
import os

app = flask.Flask(__name__, template_folder = 'templates')

@app.route('/', methods = ['POST', 'GET'])

@app.route('/index', methods = ['POST', 'GET'])

def main():
    print('Мой путь:', os.path.dirname(__file__))
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        loaded_model_Depth = load_model(os.path.join(os.path.dirname(__file__),'models', 'model_Depth.h5'))
        loaded_model_Width = load_model(os.path.join(os.path.dirname(__file__),'models', 'model_Width.h5'))
        IW = float(flask.request.form['IW'])
        IF = float(flask.request.form['IF'])
        VW = float(flask.request.form['VW'])
        FP = float(flask.request.form['FP'])

        y_pred_Depth = loaded_model_Depth.predict([[IW, IF, VW, FP]])
        y_pred_Width = loaded_model_Width.predict([[IW, IF, VW, FP]])

        return render_template('main.html', result_depth = y_pred_Depth, result_width = y_pred_Width)

if __name__ == '__main__':
    app.run()