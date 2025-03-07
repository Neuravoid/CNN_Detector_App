FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/model
COPY model/final_model.pth /app/model/

COPY app/ /app/app/
COPY streamlit_app.py /app/

EXPOSE 8000 8501

CMD ["sh", "-c", "uvicorn app.app:app --host 0.0.0.0 --port 8000 & streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0"]
