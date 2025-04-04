# Use the official Python image
FROM python:3.11-slim

# Upgrade pip and install essential build tools
RUN pip install --upgrade pip setuptools wheel Cython

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the port your Flask app runs on
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
