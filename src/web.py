from flask import Flask, jsonify, request, render_template
import pickle
from flask_cors import CORS, cross_origin
app = Flask('prediction')
cors = CORS(app)
@app.route('/', methods=['GET'])
def get_prediction():
    newlabel = pickle.load(open("./data/shimmer_prediction.pickle", 'rb'))
    return render_template('index.html', activity=newlabel)


app.run(host='0.0.0.0')