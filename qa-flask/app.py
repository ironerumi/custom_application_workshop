import json
import logging
import os
import time

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request
from werkzeug.middleware.proxy_fix import ProxyFix

logging.basicConfig(
    # filename="app.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("app")
load_dotenv()

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1, x_host=1)


def get_response(promptText: str):
    starttime = time.time()
    logger.info("get_response method called with promptText: %s", promptText)
    url_format = os.environ.get("API_URL")
    deployment_id = os.environ.get("DEPLOYMENT_ID")
    datarobot_key = os.environ.get("DATAROBOT_KEY")
    datarobot_api_key = os.environ.get("API_KEY")
    url = url_format.format(deployment_id=deployment_id)

    payload = json.dumps([{"promptText": promptText}])
    headers = {
        "DataRobot-Key": datarobot_key,
        "Content-Type": "application/json; charset=UTF-8",
        "Authorization": f"Bearer {datarobot_api_key}",
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    processtime = time.time() - starttime
    logger.info("get_response method received result: %s", response.json())

    if response.status_code == 200:
        prediction = response.json()["data"][0]["prediction"]
        return {"responseText": prediction, "processTime": processtime}
    else:
        return {
            "responseText": f"Error: {response.status_code} - {response.reason}",
            "processTime": processtime,
        }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def send_information():
    logger.info("send_information method called")
    promptText = request.form.get("promptText")
    result = get_response(promptText)
    return jsonify(result)


# @app.before_request
# def add_trailing():
#     logger.info("add_trailing method called for request path: %s", request.path)
#     rp = request.path
#     if not rp.endswith("/") and "response" not in rp:
#         return redirect(rp + "/")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False, port=8080)
