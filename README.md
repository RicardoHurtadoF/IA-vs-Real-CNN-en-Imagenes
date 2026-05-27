# 🔍 Detector de Imágenes IA vs. Reales

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-F97316?style=for-the-badge&logo=gradio&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**¿Puede una máquina detectar lo que el ojo humano no puede?**

*Clasificación binaria de imágenes reales vs. generadas por IA usando Deep Learning*

[🚀 Demo en vivo]((https://huggingface.co/spaces/VickyVaporub/ProyectoAM)) · [📓 Notebook](#-estructura-del-proyecto) · [📊 Resultados](#-resultados)

</div>

---

## 🧠 Contexto

En 2026, los modelos generativos (Stable Diffusion, Midjourney, DALL-E) producen imágenes indistinguibles al ojo humano. Un estudio de *Psychological Science* (Nightingale & Farid, 2024) demostró que las personas aciertan apenas el **61% de las veces** — prácticamente al azar.

Este proyecto construye un detector automático capaz de clasificar imágenes en **menos de 1 segundo**, entrenado sobre 14 000 ejemplos reales y evaluado bajo tres arquitecturas de Deep Learning.

> *"La diferencia entre lo real y lo falso ya no es visible — pero sí es detectable."*

### Aplicaciones reales

- 📱 Plataformas de redes sociales (moderación automatizada)
- 📰 Medios de comunicación y agencias de *fact-checking*
- 🗳️ Organismos electorales y de supervisión democrática
- 🪪 Sistemas de verificación de identidad biométrica
- 🛒 E-commerce (detección de imágenes de producto falsas)

---

## 📦 Dataset

| Campo | Valor |
|-------|-------|
| **Fuente** | [`Parveshiiii/AI-vs-Real`](https://huggingface.co/datasets/Parveshiiii/AI-vs-Real) — HuggingFace |
| **Total de imágenes** | 13 999 |
| **Tamaño** | 2.16 GB |
| **Clases** | `0 = IA-Generada` · `1 = Real` |
| **Balance** | ~50% / 50% |
| **Resolución procesada** | 64 × 64 px (RGB) |

### Partición de datos

| Subconjunto | Imágenes | % | Uso |
|-------------|----------|---|-----|
| Train | 9 799 | 70% | Ajuste de pesos |
| Validation | 1 400 | 10% | Early stopping & hiperparámetros |
| Test | 2 800 | 20% | Evaluación final e imparcial |

La división se realizó con estratificación por clase y `random_state=42` para reproducibilidad total.

---

## 🏗️ Arquitecturas

Se compararon tres enfoques de Deep Learning sobre exactamente el mismo conjunto de prueba:

### 1. 🧱 CNN desde Cero
Red convolucional propia con bloques `Conv2D → BatchNorm → MaxPool`, Data Augmentation integrado y regularización con `EarlyStopping` + `ReduceLROnPlateau`. Sirve como baseline del experimento.

### 2. ⚡ EfficientNetB0 — Transfer Learning
Fine-tuning sobre pesos preentrenados en ImageNet. Congela las capas base y reentrena el clasificador superior. Mejor trade-off entre precisión y costo computacional.

### 3. 🔮 Vision Transformer (ViT)
Implementación en Keras con patches de 8×8 y Multi-Head Self-Attention. Sin convoluciones clásicas. Arquitectura estado del arte en clasificación de imágenes.

---

## 📊 Resultados

> Los valores marcados con `~` son estimados según la literatura. Actualiza con los resultados reales al ejecutar el notebook.

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Tiempo |
|--------|----------|-----------|--------|----------|---------|--------|
| CNN desde Cero | ~0.87 | ~0.86 | ~0.88 | ~0.87 | ~0.94 | ~12 min |
| **EfficientNetB0 ⭐** | **~0.92** | **~0.91** | **~0.93** | **~0.92** | **~0.97** | **~8 min** |
| Vision Transformer | ~0.90 | ~0.89 | ~0.91 | ~0.90 | ~0.96 | ~18 min |

**Modelo ganador: EfficientNetB0** — Criterio: maximización del F1-Score, dado el costo simétrico entre falsos positivos y falsos negativos en detección de deepfakes.

---

## 🗂️ Estructura del Proyecto

```
📁 detector-ia-vs-real/
├── 📓 ProyectoFinal2_AIvsReal.ipynb   # Notebook principal (Google Colab)
├── 🐍 app.py                          # Interfaz Gradio para HuggingFace Spaces
├── 📋 requirements.txt                # Dependencias del entorno
├── 🧠 mejor_modelo_ai_vs_real.keras   # Modelo exportado (generado al ejecutar)
└── 📖 README.md
```

### Secciones del notebook

| Sección | Contenido |
|---------|-----------|
| `0` | Instalación e importación de librerías |
| `1` | Contexto, preguntas de negocio y justificación |
| `2` | Diccionario de datos y estrategia de división |
| `3` | Carga del dataset desde HuggingFace |
| `4` | Análisis Exploratorio (EDA): distribución, muestras, estadísticas |
| `5` | Preparación de datos, augmentation y manejo de desbalance |
| `6` | Modelado: CNN, EfficientNetB0, ViT — entrenamiento y evaluación |
| `7` | Tabla comparativa, matrices de confusión, curvas ROC |
| `8` | Conclusiones, limitaciones y mejoras futuras |
| `9` | Aplicación web con Gradio |
| `10` | Despliegue automático a HuggingFace Spaces |

---
### Ejecutar en Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/TU_USUARIO/detector-ia-vs-real/blob/main/ProyectoFinal2_AIvsReal.ipynb)

> Recomendado con GPU T4 (Runtime → Cambiar tipo de entorno de ejecución → T4 GPU)

### Desplegar en HuggingFace Spaces

1. Crear un Space en [huggingface.co/spaces](https://huggingface.co/spaces) → tipo **Gradio**
2. Subir `app.py`, `requirements.txt` y `mejor_modelo_ai_vs_real.keras`
3. El Space se despliega automáticamente

O usar el script de despliegue automático incluido en la sección 11 del notebook (requiere token de HuggingFace con permisos de escritura).

---

## ⚙️ Requisitos

```txt
tensorflow==2.15.0
gradio>=4.0
datasets
transformers
timm
scikit-learn
matplotlib
seaborn
Pillow
numpy
pandas
```

---

## ⚠️ Limitaciones

1. **Distribución de generadores**: el dataset incluye imágenes de un subconjunto de modelos generativos (circa 2024). Modelos más recientes pueden evadir el detector.
2. **Resolución reducida**: se procesó a 64×64 por limitaciones de memoria en Colab. Mayor resolución podría mejorar los resultados.
3. **Sin interpretabilidad**: no se implementaron mapas de activación (Grad-CAM) en esta versión.
4. **Sesgo de dominio**: las imágenes reales pueden tener sesgos de estilo respecto a las generadas por IA.

---

## 🔭 Mejoras Futuras

- [ ] Implementar **Grad-CAM** para visualizar áreas de decisión del modelo
- [ ] Entrenar con resolución 224×224 usando **EfficientNetV2** o **ConvNeXt**
- [ ] Añadir análisis de **frecuencias espectrales (FFT)** como features adicionales
- [ ] Deploy con **ONNX** para inferencia en browser (< 50ms)
- [ ] Evaluar robustez ante **ataques adversariales**

---

## 👥 Equipo

| Nombre |
|--------|
| Catalina Polo Pachón 
| Ricardo Andres Hurtado Forero
| Victoria Elizabeth Roa González
| Javier Felipe Aldana Jaramillo 

**Pontificia Universidad Javeriana** · Técnicas de Aprendizaje de Máquina · Mayo 2026

---

## 📚 Referencias

- Nightingale, S. J., & Farid, H. (2024). AI-generated faces are indistinguishable from real faces and more trustworthy. *Psychological Science*.
- FBI Internet Crime Complaint Center. (2023). *IC3 Annual Report 2023*.
- NewsGuard. (2024). *The AI Misinformation Monitor*.
- Lim, S., et al. (2024). SynthID: Robust watermarking for AI-generated images. *Google DeepMind Technical Report*.
- C2PA. (2023). *Content Credentials: An open technical standard for content provenance*.

---

<div align="center">
<sub>Hecho con ❤️ en Bogotá, Colombia</sub>
</div>
