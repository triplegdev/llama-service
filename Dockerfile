FROM python:3.9-slim

# Install dependencies
RUN pip install torch transformers

# Copy model files
COPY llama-small /app/llama-small

# Set working directory
WORKDIR /app

# Run command
CMD ["python", "app.py"]
