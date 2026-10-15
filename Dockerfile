FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pristine copies of the notebooks inside the image; the compose file mounts
# ./notebooks over this at runtime so edits on the host win.
COPY notebooks/ ./notebooks/

EXPOSE 8888
