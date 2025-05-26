FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy code
COPY . .

# Make test script executable
RUN chmod +x test.sh

# Set entrypoint
ENTRYPOINT ["./test.sh"] 