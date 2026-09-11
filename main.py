from flask import Flask, render_template, request
import joblib
import pandas as pd


app=Flask(__name__)

model=joblib.load('boston_model.pkl')
scaler=joblib.load('scaler.pkl')


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "CRIM": [float(request.form["CRIM"])],
        "ZN": [float(request.form["ZN"])],
        "INDUS": [float(request.form["INDUS"])],
        "CHAS": [float(request.form["CHAS"])],
        "NOX": [float(request.form["NOX"])],
        "RM": [float(request.form["RM"])],
        "AGE": [float(request.form["AGE"])],
        "DIS": [float(request.form["DIS"])],
        "RAD": [float(request.form["RAD"])],
        "TAX": [float(request.form["TAX"])],
        "PTRATIO": [float(request.form["PTRATIO"])],
        "B": [float(request.form["B"])],
        "LSTAT": [float(request.form["LSTAT"])]
    }

    new_data=pd.DataFrame(data)

    new_data_scaled=scaler.transform(new_data)

    new_data_pred=model.predict(new_data_scaled)

    return render_template("index.html",new_data_pred=new_data_pred)

if __name__ == "__main__":
    app.run(debug=True)