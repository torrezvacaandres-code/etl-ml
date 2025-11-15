FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set default environment variables
ENV DATABASE_URL=''
ENV DIAS_A_PREDECIR=7
ENV RUN_SCHEDULE=02:00
ENV RUN_ONCE=false

# Set Python to run in unbuffered mode
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "etl_ml.py"]
