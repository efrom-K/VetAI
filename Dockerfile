# Используем образ с GPU
FROM tensorflow/tensorflow:2.11.0-gpu

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Исправленная установка библиотек
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    tensorflow==2.11.0 \
    numpy==1.24.4 \
    pandas==2.0.3 \
    scikit-learn==1.3.2 \
    joblib==1.3.2 \
    matplotlib==3.7.5 \
    shap==0.41.0 \
    streamlit \
    fastapi \
    uvicorn \
    pydantic \
    tqdm \
    fpdf

# Копируем проект
COPY . /app

# Открываем порты
EXPOSE 8501 8000