import pickle
from flask import Flask,request,app,jsonify,url_for,render_template


import numpy as np
import pandas as pd


app=Flask(__name__)

#load the model
regmodel=pickle.load(open('classifier.pkl','rb'))
scalar=pickle.load(open('scaler.pkl','rb'))


@app.route('/')

def home():

    return render_template('home.html')

@app.route('/predict1',methods=['POST'])
def predict1():
    return render_template('predict.html')


@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]

    final_input=scalar.transform(np.array(data).reshape(1,-1))

    print(final_input)

    output=regmodel.predict(final_input)[0]
    if (output==1):
        output="Approved"
    else:
        output="Not Approved"

    return render_template("predict.html",prediction_text="The Loan apllication is : {}".format(output))




if __name__=='__main__':
    app.run(debug=True)