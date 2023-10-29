from flask import Flask, request, jsonify
from flask_cors import CORS
from faiss_search_requisite import search_k_nearest

app = Flask(__name__)
CORS(app)


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    results = search_k_nearest(query, 10)
    return results



if __name__ == '__main__':
    app.run(port=5100, debug=True)

























