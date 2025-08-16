import pickle
# scaler = pickle.load('scaler.pkl')

from flask import Flask, request, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl','rb'))
scaler=pickle.load(open('scaling.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return jsonyfy(output[0])

# @app.route('/predict',methods=['POST'])
# def predict():
#     data=[float(x) for x in request.form.values()]
#     final_input = scaler.transform(np.array(data).reshape(1,-1))
#     print(final_input)
#     output=regmodel.predict(final_input)[0]
#     return render_template("home.html",prediction_test="The House Price Prediction Is {}".format(output))


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    features = [
        float(request.form['MedInc']),
        float(request.form['HouseAge']),
        float(request.form['AveRooms']),
        float(request.form['AveBedrms']),
        float(request.form['Population']),
        float(request.form['AveOccup']),
        float(request.form['Latitude']),
        float(request.form['Longitude'])
    ]
    
    # Transform features
    final_features = scaler.transform(np.array(features).reshape(1, -1))  # Fixed variable name
    
    # Make prediction
    prediction = regmodel.predict(final_features)
    
    return render_template('home.html', prediction=prediction)

 
if __name__=="__main__":
    app.run(debug=True)

