account_id=$(aws sts get-caller-identity --query Account --output text)
# echo "$account_id"
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $account_id.dkr.ecr.us-west-2.amazonaws.com
docker build --platform linux/amd64 -t knowledge-base-lambda-parallel:test .
docker tag knowledge-base-lambda-parallel:test $account_id.dkr.ecr.us-west-2.amazonaws.com/knowledge-base:latest
docker push $account_id.dkr.ecr.us-west-2.amazonaws.com/knowledge-base:latest
