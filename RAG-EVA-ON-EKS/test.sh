#!/bin/bash
TENANTS="tenanta tenantb"
for t in $TENANTS
do
  export TENANT=${t}
  echo "S3 access policy for ${t}rag_and_model_eva"
  envsubst < iam/s3-rag_and_model_eva-access-policy.json | \
  xargs -0 -I {} aws iam create-policy \
  --policy-name s3-rag_and_model_eva-${t} \
  --policy-document {}
done