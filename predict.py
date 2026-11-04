from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import uvicorn
import numpy as np
from typing import Literal

class Student(BaseModel):

    gender: Literal["female", "male"]
    family_background: Literal["middle_class", "lower_middle_class", "rich_class"]
    relationship_status: Literal["single", "in_a_relationship", "complicated"]
    marital_status: Literal["unmarried", "married", "separated"]
    living_style: Literal["bachelor", "family"]
    friend_circle: Literal["yes", "no"]
    home_division: Literal[
        "dhaka", "khulna", "rajshahi", "sylhet", 
        "chittagong", "rangpur", "barisal", "mymensingh"
    ]
    
    current_semester: int = Field(..., ge=0, le=13, description="Semester ranging from 0 to 13")
    age: int = Field(..., ge=17, le=100, description="Age in years")
    cgpa: float = Field(..., ge=0.0, le=4.0, description="CGPA on a 4.0 scale")
    cost_without_study: int = Field(..., ge=0, description="Monthly/Annual cost in local currency")

class PredictResponse(BaseModel):
    depression_probability: float
    depression: bool    

with open('rf_model.bin', 'rb') as f_in:
    model = pickle.load(f_in)

app = FastAPI(title="Student Depression Prediction API")

def prepare_features(student_dict):
    processed = student_dict.copy()
    
    for key, value in processed.items():
        if isinstance(value, str):
            processed[key] = value.lower().replace(' ', '_')
            
    if 'cost_without_study' in processed:
        processed['cost_without_study_log'] = np.log1p(processed['cost_without_study'])
        del processed['cost_without_study']
        
    return processed


def predict_single(student):
    result=model.predict_proba(student)[0,1]
    return float(result)


@app.post("/predict")
def predict_depression(student: Student) -> PredictResponse:
    prob=predict_single(student.model_dump())
    return PredictResponse(
        depression_probability=prob,
        depression=bool(prob>=0.25)
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)