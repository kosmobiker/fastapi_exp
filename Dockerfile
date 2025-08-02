# Use an official Python runtime as a parent image
FROM python:3.13-slim

# Set the working directory in the container
WORKDIR /app

# Copy the entire application to the container
COPY . .

# Install any needed packages specified in pyproject.toml
RUN pip install --no-cache-dir .[dev]

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application
CMD ["uvicorn", "fastapi_exp.main:app", "--host", "0.0.0.0", "--port", "8000"]