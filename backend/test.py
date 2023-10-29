import requests

def search_server(content):
    url = "http://localhost:5000/search"
    params = {"query": content}
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.text
    else:
        response.raise_for_status()