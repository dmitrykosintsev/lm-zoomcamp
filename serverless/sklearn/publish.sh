ECR_URL="387546586013.dkr.ecr.eu-west-1.amazonaws.com"
REPO_URL=${ECR_URL}/churn-prediction-lambda
REMOTE_IMAGE_TAG="${REPO_URL}:v1"

LOCAL_IMAGE=churn-prediction-lambda

docker build -t ${LOCAL_IMAGE}

aws ecr get-login-password \
  --region "eu-west-1" \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}


docker build -t churn-prediction-lambda .
docker tag churn-prediction-lambda ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}

echo "Done"