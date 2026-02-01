FROM python:3.11-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p data/logs storage

# Railway inject PORT เอง ไม่ต้อง set ที่นี่
EXPOSE 8080

CMD ["python", "-m", "app.main"]