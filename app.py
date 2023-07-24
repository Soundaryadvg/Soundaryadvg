from flask import Flask;
from flask import render_template,request
import numpy as np
import joblib


app=Flask(__name__)

@app.route('/predict',methods=['POST','GET'])
def predict():
    data=[]

    for item in request.form.values():
        data.append(float(item))

    testdata=np.array([data])

    print(testdata)
    model=joblib.load('fetalmodel.pkl')
    res=model.predict(testdata)

    return render_template('index.html',p=res[0])
@app.route('/')
def index():

    return render_template('index.html')

if __name__ =='__main__':

    app.run(debug=True,host="0.0.0.0")
