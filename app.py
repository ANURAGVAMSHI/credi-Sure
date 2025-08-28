from flask import Flask, render_template, request
import numpy as np
import joblib
app=Flask(__name__)
model = joblib.load("anurag_model.pkl") 
encoder=joblib.load("anurag_encoder.pkl")
@app.route('/')
def home ():
    return render_template('index.html')# write here html page
@app.route('/predict',methods = ['POST'])
def predict():
    try:
        input_dict={ 
            "Gender" : request.form["Gender"],
            "Married": request.form["Married"],
            "Dependents": request.form["Dependents"],
            "Education": request.form["Education"],
            "Self_Employed": request.form["Self_Employed"],
            "ApplicantIncome": float(request.form["ApplicantIncome"]),
            "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
            "LoanAmount": float(request.form["LoanAmount"]),
            "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
            "Property_Area": request.form["Property_Area"]
            }
        
        for col in encoder:
            if col in input_dict:
                input_dict [col] = encoder[col].transform([input_dict[col]])[0]

        features = np.array(list(input_dict.values())).reshape(1, -1)
        prediction=model.predict(features)[0]
        
        return render_template('index.html', prediction_text="Loan Status: " + ("YES" if prediction == 1 else "NO" ))
    except Exception as e:
        return render_template('index.html', prediction_text="Error: " + str (e))
        
if __name__ == '__main__':
    app.run(debug = True) 
        