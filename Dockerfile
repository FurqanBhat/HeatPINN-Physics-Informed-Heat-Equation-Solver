FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first for optimized caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project into the container
COPY src ./src
COPY outputs ./outputs

# Default command (can be overridden)
CMD ["python", "src/main.py"]
