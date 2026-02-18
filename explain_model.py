import pandas as pd
import numpy as np
import joblib
import shap
import tensorflow as tf
import matplotlib.pyplot as plt

# --- CONFIG ---
MODEL_PATH = 'full_neural_network_model_v15_opt.h5'
FEATURES_PATH = 'full_feature_names_v15.pkl'
PREPROCESSOR_PATH = 'full_preprocessor_v15.pkl'
DATASET_PATH = 'simulated_data_1500k_v1.3.csv' 

def main():
    # 1. Загрузка
    print("Загрузка модели и данных...")
    model = tf.keras.models.load_model(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    scaler = joblib.load(PREPROCESSOR_PATH)
    
    # Берем небольшую выборку для объяснения (SHAP очень ресурсозатратный)
    df = pd.read_csv(DATASET_PATH, nrows=500)
    X = scaler.transform(df[feature_names]).astype(np.float32)

    # 2. Инициализация Explainer
    # DeepExplainer оптимизирован для нейросетей
    # Используем медиану как фоновое распределение
    background = X[np.random.choice(X.shape[0], 100, replace=False)]
    explainer = shap.DeepExplainer(model, background)
    
    # 3. Расчет
    print("Расчет SHAP значений (может занять 1-2 минуты)...")
    shap_values = explainer.shap_values(X[:100]) # Считаем для первых 100 записей

    # 4. Визуализация
    print("Сохранение графиков...")
    plt.figure(figsize=(12, 8))
    # Summary plot показывает общую важность признаков
    shap.summary_plot(shap_values, X[:100], feature_names=feature_names, show=False)
    plt.savefig('shap_summary_v15.png')
    plt.close()
    
    print("✅ Готово. Результат в shap_summary_v15.png")

if __name__ == "__main__":
    main()