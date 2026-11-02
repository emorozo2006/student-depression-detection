import pickle
import numpy as np
from flask import Flask
from flask import request
from flask import jsonify

model_file='rf_model.bin'

with open(model_file, 'rb') as f_in:
    dv, scaler, model=pickle.load(f_in)

app=Flask('depression')

@app.route('/predict', methods=['POST'])
def predict():
    student = request.get_json()

    str_columns=[
       'gender', 
       'family_background', 
       'relationship_status',
       'marital_status',
       'living_style', 
       'friend_circle', 
       'home_division'
       ]
    
    for c in str_columns:
        if c in student and isinstance(student[c], str):
            student[c]=student[c].lower().replace(' ', '_')
    cost=student.get('cost_without_study', 0)
    student['cost_without_study_log']=np.log1p(cost)

    numerical = ['age', 'cgpa', 'cost_without_study_log']
    
    num_values = [[
        student.get('age', 0), 
        student.get('cgpa', 0.0), 
        student['cost_without_study_log']
    ]]

    scaled_values = scaler.transform(num_values)[0]

    student['age'] = scaled_values[0]
    student['cgpa'] = scaled_values[1]
    student['cost_without_study_log'] = scaled_values[2]

    X = dv.transform([student])
    y_pred = model.predict_proba(X)[0, 1]
    depression = y_pred >= 0.2

    result = {
        'depression_probability': float(y_pred),
        'depression': bool(depression)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)