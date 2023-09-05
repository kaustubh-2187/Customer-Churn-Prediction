from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open("model.pkl","rb"))
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_churn():
    age = int(request.form.get('age'))
    gender = request.form.get('gender')
    sub_len_months = int(request.form.get('sub_len_months'))
    location = request.form.get('location')
    monthly_bill = float(request.form.get('monthly_bill'))
    total_usage = int(request.form.get('total_usage'))

    if gender=="male":
        gender=2
    else:
        gender=1

    location_keys = {
        "Houston" : 2,
        "Los Angeles" : 3,
        "Miami" : 4,
        "Chicago" : 1,
        "New York" : 5
    }

    if location not in location_keys.keys():
        location=6
    else:
        location = location_keys[location]

    data = np.array([age, gender, location, sub_len_months, monthly_bill, total_usage])

    # Reshape the array into a 2D array with a single row
    result = model.predict(data.reshape(1, -1))

    if result[0] == 1:
        result = "Churned"
    else:
        result = "Not Churned"

    return render_template("index.html", result=result)

if __name__== '__main__':
    app.run(host="0.0.0.0", port=8080)
