from flask import Flask, request, jsonify
from flask_cors import CORS
from faiss_search_requisite import search_k_nearest

app = Flask(__name__)
CORS(app)


@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    K = int(request.args.get('K') or "3")
  
    results = search_k_nearest(query, K)
    return results



if __name__ == '__main__':
    app.run(port=5100, debug=True)

























