"""
Datos para pruebas:

Persona que sobrevivió y embarcó en C
{
  "Pclass": 1,
  "Sex": "female",
  "Age": 29,
  "SibSp": 0,
  "Parch": 0,
  "Fare": 71.2833,
  "Embarked": "C"
}

Persona que no sobrevivió y embarcó en S
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.925,
  "Embarked": "S"
}

Persona que sobrevivió y embarcó en Q
{
  "Pclass": 2,
  "Sex": "female",
  "Age": 38,
  "SibSp": 1,
  "Parch": 1,
  "Fare": 71.2833,
  "Embarked": "Q"
}

"""

from fastapi import FastAPI, HTTPException, Header, Depends
import joblib
import pandas as pd
import traceback

model = joblib.load('titanic_model.sav')

app = FastAPI()

API_KEY = 'MIOTI_2024'  

def verify_api_key(api_key: str = Header(None)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

def data_prep(message: dict):
    sex_mapping = {'male': 0, 'female': 1}
    embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
    
    message['Sex'] = sex_mapping.get(message.get('Sex', '').lower(), 0)
    message['Embarked'] = embarked_mapping.get(message.get('Embarked', '').upper(), 0)
 
    data = pd.DataFrame([message], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
    
    return data

def survival_prediction(message: dict):
    try:
        data = data_prep(message)
        print(f"Data for prediction: {data}")
        label = model.predict(data)[0]
        proba = model.predict_proba(data)[0].max()
        return {'label': int(label), 'probability': float(proba)}
    except Exception as e:
        print(f"Error en la predicción: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {str(e)}")

@app.get('/')
def main():
    return {'message': 'Hola'}

@app.post('/survival-prediction/')
def predict_survival(message: dict, api_key: str = Depends(verify_api_key)):
    required_keys = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    for key in required_keys:
        if key not in message:
            raise HTTPException(status_code=400, detail=f"Missing {key} in request")
    
    return survival_prediction(message)
