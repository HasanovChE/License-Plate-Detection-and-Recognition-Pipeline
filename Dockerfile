# Yüngül Python versiyasından istifadə edirik
FROM python:3.9-slim

# Ətraf mühit dəyişənlərinin təyini
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# OpenCV və Tesseract üçün lazımi sistem paketlərinin quraşdırılması
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# İş qovluğunun yaradılması
WORKDIR /app

# Asılılıqların (Python paketlərinin) kopyalanması və yüklənməsi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layihə kodlarının və modelin konteynerə kopyalanması
COPY . .

# Xarici mühitə açılacaq port
EXPOSE 8000

# FastAPI serverinin işə salınması
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]