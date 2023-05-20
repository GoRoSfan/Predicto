from flask import Flask, request

from predict_model import PredictoModel

app = Flask(__name__)


@app.route('/predict')
def predict_success():
    title = request.args.get("title")
    PM = PredictoModel()
    predicted_claps = PM.predict_claps(title)
    return {
        "predicted_claps": predicted_claps
    }


if __name__ == '__main__':
    app.run()
