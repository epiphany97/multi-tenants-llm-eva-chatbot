aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 682024804674.dkr.ecr.us-west-2.amazonaws.com
docker build --platform linux/amd64 -t knowledge-base-lambda-parallel:test .
docker tag knowledge-base-lambda-parallel:test 682024804674.dkr.ecr.us-west-2.amazonaws.com/knowledge-base:latest
docker push 682024804674.dkr.ecr.us-west-2.amazonaws.com/knowledge-base:latest