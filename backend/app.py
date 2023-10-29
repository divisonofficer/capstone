SEARCH_BACKEND = "http://localhost:5100"
from flask_cors import CORS
import os
import json

from flask import Flask, request, Response, send_from_directory
import requests
app = Flask(__name__)
CORS(app)

@app.route("/", methods=["GET"])
def index():
    #send static fild ../frontend/index.html
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    frontend_directory = os.path.join(parent_directory, 'frontend')
    return send_from_directory(frontend_directory, "index.html")

@app.route("/function.js", methods=["GET"])
def index_js():
    return static_dir("function.js")
@app.route("/style.css", methods=["GET"])
def index_css():
    return static_dir("style.css")

def static_dir(path):
    #send static files in ../frontend/static
    parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    frontend_directory = os.path.join(parent_directory, 'frontend')
    return send_from_directory(frontend_directory, path)

@app.route("/search", methods=["GET"])
def search():
    # send get request to :5100/search and return response
    query = request.args.get('query')
    K = request.args.get('K') or 5
    url = SEARCH_BACKEND + "/search"
    params = {"query": query, "K" : K}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        body =  json.dumps(response.json(), ensure_ascii=False)
        return Response(body, content_type="application/json; charset=utf-8")
    else:
        response.raise_for_status()









if __name__ == "__main__":
    app.run(port = 5001, debug=True)