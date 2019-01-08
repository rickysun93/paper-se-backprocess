from flask import Flask, request, jsonify
import flask_cors as fcs

from services import paperservice

app = Flask(__name__)
fcs.CORS(app, supports_credentials=True)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/search')
def search():
    searchtype = request.args.get('searchtype', '')
    query = request.args.get('query', '')
    region = request.args.get('region', '')
    if searchtype == 'paper':
        return jsonify(paperservice.papersearch(query, region))


@app.route('/searchid')
def searchid():
    query = request.args.get('query', '')
    return jsonify(paperservice.paperidsearch(query))


if __name__ == '__main__':
    app.run()
