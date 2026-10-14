"""Flask interface - one page with a tab per model, plus a small JSON API.

Author: Ahmed Shahzad
Developer: Ahmed Shahzad

Run:  python flask_app.py      then open http://127.0.0.1:5000
API:  GET  /api/info
      POST /api/predict/<maternal|pulsar>   body: {"Feature": value, ...}
"""
from flask import Flask, jsonify, render_template, request

from common import META, get_datasets, predict, warm_up

app = Flask(__name__)
warm_up()  # load the models now, not on the first request


@app.get("/")
def index():
    return render_template("index.html", datasets=get_datasets())


@app.get("/api/info")
def api_info():
    return jsonify(get_datasets())


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/api/predict/<key>")
def api_predict(key):
    if key not in META:
        return jsonify(error=f"Unknown dataset '{key}'"), 404

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = request.form.to_dict()

    try:
        out = predict(key, data)
    except ValueError as err:
        return jsonify(error=str(err)), 400

    return jsonify(dataset=key, label=META[key]["target_label"], **out)


if __name__ == "__main__":
    app.run(debug=True)
