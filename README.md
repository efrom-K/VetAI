# 🐾 VetAI | Clinical Decision Support System (CDSS)

**Official Version:** `0.4.1-alpha` | **Neural Kernel:** `v15_opt (TensorFlow-Powered/Optimized)`  
**Academic Partner:** [Rosbiotech University] | **Project Status:** University Startup

---

## 📋 Overview
**VetAI** is a high-tech ecosystem for predictive analytics in veterinary medicine. We are developing a "smart core" trained on millions of synthesized and real clinical cases to provide veterinarians with a precise diagnostic vector.

The **Neural Kernel v15_opt** utilizes a hybrid architecture: a combination of Deep Learning neural networks based on TensorFlow and gradient boosting methods for tabular data interpretation.

---

## 👥 Core Team & Authorship

The VetAI project is the result of a synergy between advanced software development and evidence-based veterinary medicine.

### **Efim D. Romanchenko**
**Role:** Chief Technology Officer (CTO) & Lead System Architect, Full-Stack Engineer, ML Architect.
* **Contribution:** Design and implementation of the full-cycle system architecture. 
* **Engineering:** Development of the v15_opt neural kernel, creation of a custom data preprocessor, design of microservice architecture (Backend/Frontend), and deployment infrastructure.
* **Expertise:** System integration, neural network optimization for high-load tasks, and Deep Learning model architecture.

### **Ilya A. Svechnikov**
**Role:** Chief Medical Officer (CMO) & Domain Expert (DVM), Veterinary Expert, Medical Data Scientist.
* **Contribution:** Scientific leadership and formation of the project's fundamental knowledge base. 
* **Data Science:** Collection, verification, and structuring of clinical data. Development of unique medical parsing algorithms based on fundamental veterinary literature.
* **Expertise:** Practicing veterinarian. Expert in differential diagnosis and dataset curation for clinical expert systems.

---

## 📊 Dataset: Expert-Curated Medical Data

One of VetAI's primary innovations is its Data-Origin. Unlike models trained on unstructured "open-source" data, our kernel is based on:

1. **Expert Synthesis:** A database developed over time by practicing clinicians.
2. **Clinical Accuracy:** Every pattern in DISEASE_PROFILES has been verified against evidence-based medicine protocols.
3. **Structured Knowledge:** Transformation of years of clinical experience into machine-readable CSV matrices, accounting for the finest nuances of pathophysiology.

---

## 🛠 Technology Stack & Infrastructure
The project is packaged in a high-performance environment to ensure stability and scalability.

### Model Environment (Dockerized)
We utilize a specialized stack for GPU acceleration, enabling inference within milliseconds.
* **Base Image:** `tensorflow/tensorflow:2.11.0-gpu`
* **Core Engine:** TensorFlow 2.11.0 + Keras
* **Explainability (XAI):** `SHAP` (SHapley Additive exPlanations) to interpret model decision-making.
* **Deployment:** Flask + Gunicorn for the API layer, Streamlit for the frontend demo.

---

## 🔬 Architecture: Neural Kernel v15
The v15 architecture is built on the Biologically Constrained ML principle.

1. **Feature Engineering & Latent Space** The model processes over 200 input parameters, including sparse categorical data (One-Hot Encoded). The use of protobuf==3.20.0 ensures fast data serialization between system components.

2. **Explainable AI (SHAP)** Argumentation is as vital as the result in veterinary medicine. SHAP integration allows for visual representation of each symptom's contribution:
   * *Example: The model highlights that "Enlarged Abdomen" added +40% to GDV probability, while "Normal Temperature" decreased infection probability by 15%.*

3. **Data Synthesis Protocol** Training is performed on a dataset of 1,500,000 records, based on proprietary medical profiles hand-curated by Ilya Svechnikov. This allows the model to identify rare pathologies that a doctor might encounter only once a decade.

---

## 💻 System Components & UI

VetAI is divided into two independent layers interacting via a RESTful API, allowing Backend scaling independently from the interface.

### 🎨 Frontend (UI Layer)
The UI is implemented as a high-load web application providing cross-platform access for clinicians.

* **Stack:** Streamlit / React-lite
* **Features:**
    * Interactive anamnesis collection with dynamic validation.
    * Real-time diagnostic vector visualization.
    * **XAI Dashboard:** Embedded SHAP value charts for diagnostic justification.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a97f57de-9f8b-4a93-9806-1886699ba07e" alt="VetAI Demo UI" width="900">
  <br>
  <i>Интерфейс системы Neural Kernel v15 в режиме диагностической сессии</i>
</p>

### ⚙️ Backend (Inference Engine)
The mathematical "heart" of the system, responsible for high-performance computing and neural kernel operations.

* **Stack:** Python 3.9+ / Flask / Gunicorn
* **Architecture:**
    * **Preprocessing Pipeline:** Automatic input normalization and sparse matrix handling.
    * **Inference Engine:** Asynchronous TensorFlow model execution using CUDA cores.
    * **API Layer:** Standardized JSON/Protobuf data exchange.
* **Optimization:** Multi-threaded Gunicorn implementation for parallel request handling.

---

## 📄 Reporting tools

An integrated veterinary protocol generation module allows for instant conversion of analysis results into print-ready documents.

### Key Features:
* **Automated Generation:** Protocols include diagnosis, calculated probability, full anamnesis, and identified symptoms.
* **Multilingual Support:** DejaVuSans fonts ensure correct Cyrillic/International text rendering.
* **Modular Architecture:** PDF logic is isolated in pdf_generator.py, allowing for:
    * UI/Report redesign without touching neural network code.
    * Scalability for various formats (A4, mobile receipts, etc.).

### Technical Export Stack:
* **FPDF2:** High-speed PDF generation library for Python.
* **Byte-streaming:** Documents are generated on-the-fly in RAM and delivered to the user without temporary server files, enhancing security and performance.

---

## 📡 RESTful API Interface

VetAI provides a standardized FastAPI-based interface implementing a "Model-as-a-Service" (MaaS) architecture.

| Method | Endpoint | Description |
|:---|:---|:---|
| **GET** | / | Health-check: service status and model version monitoring. |
| **POST** | /predict | Inference: symptom vector processing and Top-N diagnosis return. |
| **GET** | /docs | Interactive Swagger UI documentation. |

### Integration Example (Payload):

```json
{
  "symptoms": {
    "Порода_собака": 1,
    "Возраст_молодой": 1,
    "Анорексия": 1,
    "Рвота_тащековая": 1
  }
}

```

---

## 🚀 Deployment & Development
### Local Setup via Docker

```bash
# Build image
docker build -t vet-ai-image . --no-cache

# Run (CPU Accelerated) APP+API
docker run -it -p 8501:8501 -p 8000:8000 --rm `
    -v "${pwd}:/app" `
    vet-ai-image `
    sh -c "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501"

# Run (GPU Accelerated) APP+API
docker run --gpus all -it -p 8501:8501 -p 8000:8000 --rm `
    -v "${pwd}:/app" `
    vet-ai-image `
    sh -c "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501"

# Run benchmark
docker run --gpus all --rm -v ${pwd}:/app vet-ai-image python benchmark.py

```

**Download the weight file from the Releases section and place it in the root folder.**

### CI/CD Pipeline
Ready for **Kubernetes (K8s)**. 

---

## 📊 Scientific Methodology

* **Precision:** `0.94+`
* **Recall:** `0.91+`
* **Core Inference Latency:** `0.0695ms` (For Nvidia 30-series)
* **End-to-End System Latency:** `58ms` 

---

## 📊 Performance & Hardware Validation

Technical verification of the v15_opt kernel shows a direct correlation between hardware and diagnostic hypothesis speed.

| Hardware Configuration | Compute module | Core Inference Latency (мс) | End-to-End System Latency (мс) | Status |
|:---|:---|:---|:---|:---|
| **Workstation:** i5-11400f / 16GB ddr4 3200/ **NVIDIA RTX 3060 12GB GDDR6** | **GPU (CUDA)** |**0.0695 ms**| **58 ms** | **Real-time:** |
| **Workstation:** i5-13400f / 32GB ddr4 3200 / **NVIDIA RTX 4060 8GB GDDR6** | **GPU (CUDA)** | **0.0423 ms**| **45 ms** | **Real-time:** |
| **Mobile Workstation:** r5-3550h / 32GB ddr4 2400 / **NVIDIA GTX 1650 4GB GDDR5** | **GPU (CUDA)** | **0.0990 ms**| **93 ms** | **Real-time:** |
| **Mobile Node:** i7-8650u / 16GB ddr4 2400 (No GPU) | **CPU (AVX2/FMA)** | **4.4691 ms**| **276 ms** | **Stable (Fallback mode):** |
| **Mobile Node:** r7-4700u / 16GB ddr4 3200 (No GPU) | **CPU (AVX2/FMA)** | **4.2431 ms**| **238 ms** | **Stable (Fallback mode):** |
| **Mobile Node:** i5-2520M / 4GB ddr3 1333 (No GPU) | **CPU (AVX)** | **25.8865 ms** | **1467 ms** | **Not stable (Fallback mode):** |

---

## 🛠 Engineering & Performance Optimization

### 1. Mixed Precision Computing

Enabled for NVIDIA Ampere/Turing architectures. Critical nodes remain in float32 for numerical stability, while tensor operations run in float16.

### 2. Runtime Efficiency

Native Linux environments demonstrate a 2.4x performance boost over WSL2 for CPU-based inference.

### 3. Serialization Refactoring

Switched to a persistence strategy that excludes optimizer graphs (include_optimizer=False) to ensure cross-version compatibility.

---

## 🖥️ System Requirements Specification

### 1. Minimum requirements

| Component | Local Station (Client) | Server (Standard Server) |
| :--- | :--- | :--- |
| **CPU** | **Intel Core i3-10100** / **Ryzen 3 3100** | **Intel Core i5-11400** / **Ryzen 5 3600** |
| **RAM** | 16 GB DDR4 | 16 GB DDR4/DDR5 |
| **GPU** | Integrated (AVX2 support) | **NVIDIA RTX 3060**  |
| **VRAM** | — | 12 GB GDDR6 |
| **Latency** | ~200 - 300 ms | ~30 - 50 ms |

### 2. Recommended requirements

| Component | Local Station (High-end) | Server (Enterprise) |
| :--- | :--- | :--- |
| **CPU** | **Intel Core i5-10400** / **Ryzen 5 5600H** | **Intel Core i7-12700** / **Ryzen 7 5800X** |
| **RAM** | **32 GB DDR4/DDR5** | **32 GB+ DDR5** |
| **GPU** | **NVIDIA GTX 1650** | **NVIDIA RTX 4070** / **RTX 3080** |
| **VRAM** | **4 GB GDDR5** | **12 GB+ GDDR6X** |
| **Latency** | ~30 - 45 ms | **< 10 ms** |

### 🛠 Software
* **Windows:** 10/11 (64-bit) + **WSL2** (Windows Subsystem for Linux).
* **Linux:** Ubuntu 20.04 LTS / 22.04 LTS (reccomended).
* **Drivers:** NVIDIA Game Ready / Studio Driver with **CUDA 11.8+** and **cuDNN**.
* **Architecture:** x86_64 (AVX2 supported).

---
### Roadmap
* **[ ]** QA refactoring
* **[ ]** system tests in clinical setup
* **[ ]** API integration tests
* **[ ]** API benchmark
---

## 📬 Contacts & Scale-up
VetAI is actively seeking partnerships with veterinary clinics for pilot implementation.

* **University Supervisor:** [Rosbiotech Team]
* **Version:** 0.4.1 Alpha (Not for clinical use yet)
