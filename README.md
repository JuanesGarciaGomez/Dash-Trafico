# Dash Tráfico Urbano

Dashboard interactivo para el análisis y predicción del volumen de tráfico urbano, desarrollado con **Dash + Plotly**.

---

## Demo

👉 [https://dash-trafico-946953513839.us-central1.run.app](https://dash-trafico-946953513839.us-central1.run.app)

---

## Despliegue en Google Cloud Run

### Requisitos previos
- Cuenta de Google Cloud con un proyecto activo
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) instalado y autenticado

### Pasos para desplegar

```bash
# 1. Autenticarse en Google Cloud
gcloud auth login

# 2. Configurar el proyecto
gcloud config set project TU_PROJECT_ID

# 3. Habilitar Cloud Run y Cloud Build
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# 4. Desplegar la app
gcloud run deploy dash-trafico \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --port 8050
```

Una vez desplegado, Google Cloud te entregará una URL del tipo:
```
https://dash-trafico-XXXXXXXXXX-uc.a.run.app
```

---

## Ejecución local

```bash
pip install -r requirements.txt
python prediccion_volumen_trafico_actualizado.py
```

Abre tu navegador en: [http://localhost:8050](http://localhost:8050)

---

## Estructura del proyecto

```
dashtraffic/
├── prediccion_volumen_trafico_actualizado.py   # App principal
├── requirements.txt                            # Dependencias
├── Procfile                                    # Configuración para Cloud Run
├── assets/                                     # Recursos estáticos
│   ├── traffic_bg.jpeg
│   ├── carretera_noche.mp4
│   └── Semaforo.mp4
└── README.md
```

---

## Tecnologías

- [Dash](https://dash.plotly.com/) — Framework para dashboards interactivos
- [Plotly](https://plotly.com/) — Visualizaciones interactivas
- [NumPy](https://numpy.org/) — Cálculos numéricos
- [Gunicorn](https://gunicorn.org/) — Servidor WSGI para producción
- [Google Cloud Run](https://cloud.google.com/run) — Despliegue serverless
