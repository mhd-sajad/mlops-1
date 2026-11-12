FROM python:3.13-slim

WORKDIR /workspace

COPY requirements.txt .

RUN pip install --no-cache -r requirements.txt

COPY . .

RUN python train.py .
RUN python train_logreg.py .

EXPOSE 8000


CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]