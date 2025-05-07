FROM python:3.11-alpine

WORKDIR /app

# Install dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .

# Create a directory for the credentials
RUN mkdir -p /app/credentials

# Expose the port
EXPOSE 7860

# Command to run the application
# Use the default Hugging Face port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
