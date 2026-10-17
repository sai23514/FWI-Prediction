FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY app ./app
COPY models/fwi_model.joblib models/fwi_weather_only.joblib ./models/
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "gunicorn --chdir app -w 2 -b 0.0.0.0:${PORT} app:app"]
