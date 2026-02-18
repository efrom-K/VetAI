from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import tensorflow as tf
import numpy as np
if not hasattr(np, 'int'):
    np.int = int
import pandas as pd
from typing import Dict, List

# --- ИНИЦИАЛИЗАЦИЯ ---
app = FastAPI(
    title="VetAI Diagnostic API",
    description="API для предварительной диагностики заболеваний животных (v0.4.1)",
    version="0.4.1"
)

# Загружаем артефакты один раз при старте сервера
MODEL_PATH = 'full_neural_network_model_v15_opt.h5'
PREPROCESSOR_PATH = 'full_preprocessor_v15.pkl'
ENCODER_PATH = 'full_label_encoder_v15.pkl'
FEATURES_PATH = 'full_feature_names_v15.pkl'

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    training_features = joblib.load(FEATURES_PATH)
except Exception as e:
    print(f"Критическая ошибка загрузки: {e}")
    # В реальности тут лучше остановить запуск, но для примера оставим так

# --- СХЕМЫ ДАННЫХ ---
class DiagnosticRequest(BaseModel):
    # Словарь симптомов, где ключ - название признака из FEATURES_PATH, значение - 1 или 0
    # Пример: {"Порода_собака": 1, "Возраст_молодой": 1, "Рвота_тащековая": 1}
    symptoms: Dict[str, int]

class DiagnosisResult(BaseModel):
    diagnosis: str
    probability: float

class DiagnosticResponse(BaseModel):
    top_diagnoses: List[DiagnosisResult]

# --- ЭНДПОИНТЫ ---

@app.get("/")
async def root():
    return {"status": "online", "model_version": "v15", "message": "VetAI API is running"}

@app.post("/predict", response_model=DiagnosticResponse)
async def predict(request: DiagnosticRequest):
    try:
        # 1. Создаем пустой DataFrame со всеми признаками
        input_df = pd.DataFrame(0, index=[0], columns=training_features)
        
        # 2. Заполняем данными из запроса
        for feature, value in request.symptoms.items():
            if feature in training_features:
                input_df.loc[0, feature] = value
        
        # 3. Препроцессинг
        processed_input = preprocessor.transform(input_df)
        
        # 4. Предсказание
        predictions_proba = model.predict(processed_input, verbose=0)[0]
        
        # 5. Берем ТОП-3 результата
        top_indices = predictions_proba.argsort()[-3:][::-1]
        
        results = []
        for idx in top_indices:
            results.append(DiagnosisResult(
                diagnosis=str(label_encoder.inverse_transform([idx])[0]),
                probability=round(float(predictions_proba[idx]), 4)
            ))
            
        return DiagnosticResponse(top_diagnoses=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск: uvicorn api:app --host 0.0.0.0 --port 8000