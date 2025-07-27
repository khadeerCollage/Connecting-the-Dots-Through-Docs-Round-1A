# 1. 
FROM python:3.10-alpine

# 2. 
WORKDIR /app

# 3. 
COPY requirements.txt .

# 4.
RUN apk add --no-cache --virtual .build-deps gcc musl-dev freetype-dev g++ \
    && apk add --no-cache freetype libstdc++ \
    && pip install --no-cache-dir -r requirements.txt \
    && apk del .build-deps \
    && rm -rf /var/cache/apk/* /root/.cache/pip/*

# 5. 
COPY app/pdf_process.py .
COPY app/inputs ./inputs
COPY app/outputs ./outputs

CMD ["python", "pdf_process.py"]
