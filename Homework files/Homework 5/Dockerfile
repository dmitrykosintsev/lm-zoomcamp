FROM agrigorev/zoomcamp-model:2025

# Set working directory (the base image uses /code)
WORKDIR /code

# Copy pyproject.toml and install dependencies
COPY pyproject.toml pyproject.toml
RUN pip install .

# Copy the prediction script
COPY predict_docker.py predict_docker.py

# Expose the port
EXPOSE 9696

# Run the FastAPI application with uvicorn
ENTRYPOINT ["uvicorn", "predict_docker:app", "--host", "0.0.0.0", "--port", "9696"]