FROM python:3.10-slim

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt


RUN python -m nltk.downloader wordnet punkt punkt_tab

COPY dreamscope_backend dreamscope_backend

# For local development, you can use the following command to run the API server:
#CMD uvicorn dreamscope_backend.api_file:app --host 0.0.0.0

# For deployment on Cloud, use the following command to run the API server:
CMD uvicorn dreamscope_backend.api_file:app --host 0.0.0.0 --port $PORT
