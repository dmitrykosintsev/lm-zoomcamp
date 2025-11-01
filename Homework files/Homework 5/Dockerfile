FROM agrigorev/zoomcamp-model:2025

# Copy pyproject.toml and install dependencies
COPY pyproject.toml pyproject.toml
RUN pip install .

# Copy the pipeline model and prediction script
COPY pipeline_v1.bin pipeline_v1.bin
COPY predict_fastapi.py predict_fastapi.py

# Expose the port
EXPOSE 9696

# Run the FastAPI application with uvicorn
ENTRYPOINT ["uvicorn", "predict_fastapi:app", "--host", "0.0.0.0", "--port", "9696"]