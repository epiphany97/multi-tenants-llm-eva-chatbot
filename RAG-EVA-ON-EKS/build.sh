#!/usr/bin/env bash
. ~/.bash_profile

echo "****Building Chatbot and RAG-API Images****"
sh image-build/build-chatbot-image.sh
docker rmi -f $(docker images -a -q) &> /dev/null
sh image-build/build-rag-api-image.sh
docker rmi -f $(docker images -a -q) &> /dev/null