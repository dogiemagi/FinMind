FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY finmind finmind
# build all artifacts (tokenizer, model, ledger, docs) at image build time
RUN python -m finmind train 400
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "finmind.api:app", "--host", "0.0.0.0", "--port", "8000"]
