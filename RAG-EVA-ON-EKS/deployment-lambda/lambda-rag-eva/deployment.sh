account_id=$(aws sts get-caller-identity --query Account --output text)
# echo "$account_id"
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $account_id.dkr.ecr.us-west-2.amazonaws.com
docker build --platform linux/amd64 -t model_eva:test .
docker tag model_eva:test $account_id.dkr.ecr.us-west-2.amazonaws.com/model_eva:latest
docker push $account_id.dkr.ecr.us-west-2.amazonaws.com/model-eva:latest
