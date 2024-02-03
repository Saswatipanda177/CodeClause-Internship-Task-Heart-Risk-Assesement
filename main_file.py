import os.path
from flask import Flask, render_template, request
import joblib
import numpy as np
import pickle

app = Flask(__name__, static_folder='static')

print("Current Working Directory:", os.getcwd())

@app.route("/")
def index():
    return render_template("home.html")


@app.route("/result", methods=['POST', 'GET'])
def result():
    gender = int(request.form['gender'])
    age = int(request.form['age'])
    hypertension = int(request.form['hypertension'])
    heart_disease = int(request.form['heart_disease'])
    ever_married = int(request.form['ever_married'])
    work_type = int(request.form['work_type'])
    Residence_type = int(request.form['Residence_type'])
    avg_glucose_level = float(request.form['avg_glucose_level'])
    bmi = float(request.form['bmi'])
    smoking_status = int(request.form['smoking_status'])

    x = np.array([gender, age, hypertension, heart_disease, ever_married, work_type, Residence_type,
                  avg_glucose_level, bmi, smoking_status]).reshape(1, -1)

    scaler_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/scaler.pkl'))
    scaler = None
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    x = scaler.transform(x)

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models/dt.sav'))
    clf = joblib.load(model_path)

    Y_pred = clf.predict(x)

    # no stroke risk
    if Y_pred == 0:
        return render_template('nostroke.html')
    else:
        return render_template('stroke.html', stage=int(Y_pred))


if __name__ == "__main__":
    app.run(debug=True, port=7384)
