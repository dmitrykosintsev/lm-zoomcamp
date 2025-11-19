# lm-zoomcamp
## Problem statement
As a teacher, I was looking into current problems that could be analysed and solved using machine learning. I came across a report from UNDP Mongolia about cyberbullying. Though I could not find any available raw data for the report, it was the most up-to-date information about any school-related issues in Mongolia. I decided to generate a synthetic dataset from the report in order to create models that could predict cyberbullying in the context of Mongolia.

My goal is to achieve such accuracy that will allow teachers and parents to use the model in order to identify potential cases of cyberbullying.

## Data & EDA insights
Script generate_data.py was used to produce the dataset. There are three outputs:
- Responses from students
- Responses from parents
- Combined dataset

I only used the first dataset, as responses from parents does not seem to add any value in identifying cyberbullying. For example, there is a huge awareness gap:

- Actual cyberbullying rate: 45.6%

- Parents who know: 3.4%

- Awareness gap: 42.2%

Therefore, I did not proceed with the parents dataset.

There were many missing values due to the fact some variables only had values when experienced_cyberbullying was True. For details, check the notebook.

## Model choice
I trained and tuned the following models:
- Linear Regression: best model with C=0.01 has AUC=0.5968 and F1 score=0.631
- Decision tree: best model with max_depth=2, min_samples_leaf=500 has AUC=0.5867 and F1 score=0.619
- Random Forest: best model with n_estimators=100, max_depth=5, min_samples_leaf=50 has AUC=0.5933 and F1 score=0.625
- GBoost: best model with eta=0.01, max_depth=4, min_child_weight=1 has AUC=0.5821 and F1 score=0.626

## Deployment
### Local Development

1. **Clone repository**
```bash
   git clone 
   cd Midterm-cyberbullying
```

2. **Create virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Train model**
```bash
   python train.py
```

5. **Run Flask app**
```bash
   python app.py
```

6. **Test prediction**
```bash
   curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{
       "age_group": "13-17",
       "gender": "Female",
       "daily_internet_hours": 6,
       "primary_activity": "Gaming",
       "uses_facebook": 1,
       "num_social_media_accounts": 3,
       "exposed_to_bad_language": 1,
       "learned_bad_words": 1,
       "received_school_education": 0,
       "awareness_level": "Low"
     }'
```

### Docker (Local)

1. **Build Docker image**
```bash
   docker build -t cyberbullying-predict:latest .
```

2. **Run container**
```bash
   docker run -p 8080:8080 cyberbullying-predict:latest
```

3. **Test**
```bash
   curl -X GET http://localhost:8080/health
```

### Google Cloud Deployment

#### Prerequisites
- Google Cloud account with billing enabled
- `gcloud` CLI installed
- Docker installed

Make sure to log in if using GCloud for the first time:
```bash
   gcloud init
```

#### Setup

1. **Create GCP Project**
```bash
   gcloud projects create cyberbullying-predict --name="Cyberbullying Prediction Model"
   gcloud config set project cyberbullying-predict
```

2. **Enable required services**
```bash
   gcloud services enable \
     cloudbuild.googleapis.com \
     run.googleapis.com \
     container.googleapis.com \
     artifactregistry.googleapis.com
```

3. **Create Artifact Registry repository**
```bash
   gcloud artifacts repositories create cyberbullying-repo \
     --repository-format=docker \
     --location=us-central1 \
     --description="Docker repository for cyberbullying prediction model"
```

4. **Configure Docker authentication**
```bash
   gcloud auth configure-docker us-central1-docker.pkg.dev
```

#### Deploy to Cloud Run

**Option A: Using gcloud (Recommended)**
```bash
# Deploy directly
gcloud run deploy cyberbullying-predict \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars MODEL_PATH=/workspace/models/model_d=5_msl=50.bin \
  --memory 1Gi \
  --cpu 1
```

**Option B: Using Cloud Build (for CI/CD)**
```bash
# Submit build
gcloud builds submit \
  --config cloud/cloudbuild.yaml \
  --substitutions _REGION=us-central1,_SERVICE_NAME=cyberbullying-predict
```

#### Get the Service URL
```bash
gcloud run services describe cyberbullying-predict \
  --region us-central1 \
  --format='value(status.url)'
```

#### Test Deployed Service
```bash
SERVICE_URL=$(gcloud run services describe cyberbullying-predict \
  --region us-central1 --format='value(status.url)')

curl -X POST ${SERVICE_URL}/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age_group": "13-17",
    "gender": "Female",
    "daily_internet_hours": 6,
    "primary_activity": "Gaming",
    "uses_facebook": 1,
    "num_social_media_accounts": 3,
    "exposed_to_bad_language": 1,
    "learned_bad_words": 1,
    "received_school_education": 0,
    "awareness_level": "Low"
  }'
```

#### View Logs
```bash
gcloud run logs read cyberbullying-predict --region us-central1
```

#### Update Model
```bash
# After updating model training
gcloud run deploy cyberbullying-predict \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## API Endpoints

### Health Check
- **Endpoint**: `GET /health`
- **Response**: `{"status": "healthy"}`

### Single Prediction
- **Endpoint**: `POST /predict`
- **Request**:
```json
  {
    "age_group": "13-17",
    "gender": "Female",
    "daily_internet_hours": 6,
    "primary_activity": "Gaming",
    "uses_facebook": 1,
    "num_social_media_accounts": 3,
    "exposed_to_bad_language": 1,
    "learned_bad_words": 1,
    "received_school_education": 0,
    "awareness_level": "Low"
  }
```
- **Response**:
```json
  {
    "prediction": {
      "cyberbullying_risk": true,
      "risk_probability": 0.65,
      "confidence": 0.65
    },
    "status": "success"
  }
```

### Batch Prediction
- **Endpoint**: `POST /predict_batch`
- **Request**: Array of student objects
- **Response**: Array of predictions

## Model Performance

- **AUC**: 0.60
- **F1 Score**: 0.63
- **Accuracy**: 0.58

## Features Used for Prediction

- age_group
- gender
- daily_internet_hours
- primary_activity
- uses_facebook
- num_social_media_accounts
- exposed_to_bad_language
- learned_bad_words
- received_school_education
- awareness_level

## Monitoring & Scaling

### View Service Metrics
```bash
gcloud run services describe cyberbullying-predict --region us-central1
```

### Scale Service
```bash
gcloud run services update cyberbullying-predict \
  --region us-central1 \
  --min-instances 1 \
  --max-instances 10
```

### Set Up Monitoring
```bash
gcloud monitoring dashboards create --config-from-file - <<EOF
{
  "displayName": "Cyberbullying Model Dashboard"
}
EOF
```

## Cleanup
```bash
# Delete Cloud Run service
gcloud run services delete cyberbullying-predict --region us-central1

# Delete Artifact Registry repository
gcloud artifacts repositories delete cyberbullying-repo --location us-central1

# Delete project (if needed)
gcloud projects delete cyberbullying-predict
```

## Troubleshooting

### Model not found
- Ensure `data/students_data.csv` is in the correct location
- Run `python train.py` locally first

### Docker build fails
- Check Python version compatibility
- Ensure all dependencies are in `requirements.txt`

### Cloud Run deployment fails
- Check gcloud authentication: `gcloud auth list`
- View build logs: `gcloud builds log <build-id>`
- Check service logs: `gcloud run logs read cyberbullying-predict`
