from flask import Flask

app = Flask(__name__)


@app.route('/<title>')
def hello_world(title):
    return {
        "message": "Sorry, but model is on development. Please wait."
    }


if __name__ == '__main__':
    app.run()
