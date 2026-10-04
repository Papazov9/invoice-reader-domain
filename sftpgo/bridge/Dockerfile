FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bridge.py .

# Reads the shared SFTPGo volume, pushes to Supabase. No inbound ports.
CMD ["python", "bridge.py"]
