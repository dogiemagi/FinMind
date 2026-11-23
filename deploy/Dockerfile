FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 DJANGO_DEBUG=0
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY manage.py .
COPY config config
COPY finmind finmind
COPY ledger ledger
COPY assistant assistant
COPY agents agents
COPY web web
# build the database, policy documents, trained model and static files at image build time
RUN python manage.py migrate --noinput \
 && python manage.py seed_ledger \
 && python manage.py train_model --steps 250 \
 && python manage.py collectstatic --noinput
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120"]
