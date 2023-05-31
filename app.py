from flask import Flask, request

from predict_model import PredictoModel, PredictoRFR

app = Flask(__name__)


@app.route('/predict/nn')
def predict_by_nn():
    title = request.args.get("title")
    PM = PredictoModel()
    predicted_claps = float(PM.predict_claps(title)[0])
    return {
        "predicted_claps": predicted_claps
    }


@app.route('/predict/rfr')
def predict_by_rfr():
    title = request.args.get("title")
    PRFR = PredictoRFR()
    predicted_claps = PRFR.predict_claps(title)
    return {
        "predicted_claps": predicted_claps
    }


if __name__ == '__main__':
    app.run()
