FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt -q
COPY app.py semantic_classifier.py preprocessing.py military_keywords.txt ./
COPY static/ static/
CMD ["python", "app.py"]
