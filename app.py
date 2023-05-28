from flask import Flask, request

from predict_model import PredictoModel, PredictoRFR

app = Flask(__name__)


@app.route('/predict/nn')
def predict_success():
    title = request.args.get("title")
    PM = PredictoModel()
    predicted_claps = PM.predict_claps(title)
    return {
        "predicted_claps": predicted_claps
    }


@app.route('/predict/rfr')
def predict_success():
    title = request.args.get("title")
    PRFR = PredictoRFR()
    predicted_claps = PRFR.predict_claps(title)
    return {
        "predicted_claps": predicted_claps
    }


if __name__ == '__main__':
    app.run()
