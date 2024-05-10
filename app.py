
# Import the required packages
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')
from fastapi import FastAPI
from pydantic import BaseModel, BaseSettings
import uvicorn

app = FastAPI()

# load the trained model
trained_model = open("ccrf_model.pkl", "rb")
model = joblib.load(trained_model)

# Create a class 'credit_default' that defines data types expected from the user.
class credit_default(BaseModel):
    Limit_bal: float
    Bill_amt1: float
    Bill_amt2: float
    Bill_amt3: float
    Bill_amt4: float
    Bill_amt5: float
    Bill_amt6: float
    Pay_amt1: float
    Pay_amt2: float
    Pay_amt3: float
    Pay_amt4: float
    Pay_amt5: float
    Pay_amt6: float
    Graduate_school: int
    High_school: int
    Others: int
    University: int
    Marriage: int
    Age20s: int
    Age30s:int
    Age40s: int
    Age50s: int
    Age60s: int
    Age70s: int
    Sex: int
    Delayed_1: int
    Delayed_2: int
    Delayed_3: int
    Delayed_4: int
    Delayed_5: int
    Delayed_6: int
    
@app.get('/')
def index():
    return{'message': 'Credit deafult app'}

@app.post('/predict')
def predict_default(customer_details:credit_default):
    data = customer_details.dict()
    limit_bal = data['Limit_bal']
    bill_amt1 = data['Bill_amt1']
    bill_amt2 = data['Bill_amt2']
    bill_amt3 = data['Bill_amt3']
    bill_amt4 = data['Bill_amt4']
    bill_amt5 = data['Bill_amt5']
    bill_amt6 = data['Bill_amt6']
    pay_amt1 = data['Pay_amt1']
    pay_amt2 = data['Pay_amt2']
    pay_amt3 = data['Pay_amt3']
    pay_amt4 = data['Pay_amt4']
    pay_amt5 = data['Pay_amt5']
    pay_amt6 = data['Pay_amt6']
    graduate_school = data['Graduate_school']
    high_school = data['High_school']
    others = data['Others']
    university = data['University']
    marriage = data['Marriage']
    age20s = data['Age20s']
    age30s = data['Age30s']
    age40s = data['Age40s']
    age50s = data['Age50s']
    age60s = data['Age60s']
    age70s = data['Age70s']
    sex = data['Sex']
    delayed_1 = data['Delayed_1']
    delayed_2 = data['Delayed_2']
    delayed_3 = data['Delayed_3']
    delayed_4 = data['Delayed_4']
    delayed_5 = data['Delayed_5']
    delayed_6 = data['Delayed_6']
    
    # Make predictions
    
    prediction = model.predict([[limit_bal, bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6, pay_amt1, 
                                 pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6, graduate_school, high_school, others, 
                                 university, marriage, sex, age20s, age30s, age40s, age50s, age60s, age70s, delayed_1, 
                                 delayed_2, delayed_3, delayed_4, delayed_5, delayed_6]])
    
    if prediction == 0:
        pred = 'Pay'
    else:
        pred = 'Default'
        
    return {'status':pred}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)

