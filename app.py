from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load("xgboost_churn_model.pkl")
scaler = joblib.load("churn_scaler.pkl")

# Features expected from form input (original dataset minus target)
FEATURES = [
    'CLIENTNUM', 'Customer_Age', 'Gender', 'Dependent_count',
    'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category',
    'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
    'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
    'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
]

@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form data and convert to DataFrame
        data = {f: request.form.get(f) for f in FEATURES}
        df = pd.DataFrame([data])

        # Drop CLIENTNUM before processing
        df.drop(columns=["CLIENTNUM"], inplace=True)

        # Apply same preprocessing steps as during training
        # Note: must include encoding steps if needed
        df_processed = pd.get_dummies(df)

        # Align with training columns
        expected_columns = scaler.feature_names_in_  # saved during scaling
        for col in expected_columns:
            if col not in df_processed:
                df_processed[col] = 0  # add missing dummies with 0
        df_processed = df_processed[expected_columns]  # reorder columns

        # Scale
        df_scaled = scaler.transform(df_processed)

        # Predict
        pred = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1]

        return render_template("result.html", prediction=pred, probability=round(prob, 4))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=8080)
