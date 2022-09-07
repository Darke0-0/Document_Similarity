# Serve model as a flask application

# Importing Libraries
# from crypt import methods
from os import system
import sys
from gensim.models.keyedvectors import KeyedVectors
from TextSim import TextSim
from flask import Flask, request

"""
Defining model
Loading the word embeddings
"""

# Initializing the Flask object to run the flask app
app = Flask(__name__, static_folder='static', template_folder='templates')

"""
This rendering template is done if it get's any GET Request
"""


@app.route('/', methods=['POST', 'GET'])
def main():
    if request.method == 'GET':
        return 'Please use POST method'
    if request.method == 'POST':
        features = request.json
        print(features)
        para1 = features["text1"]
        para2 = features["text2"]
        result = model.calculate_similarity(para1, para2)
        return result

if __name__ == '__main__':
    # We need to run the app to run the server
    app.run(port=5000)
