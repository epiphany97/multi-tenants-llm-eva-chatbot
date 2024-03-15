FROM public.ecr.aws/lambda/python:3.10

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy function code
# COPY ["lambda_function.py", "generate_response.py", "retreival.py", "${LAMBDA_TASK_ROOT}"]
COPY ["lambda_function_parallel.py", "generate_response.py", "retreival.py", "${LAMBDA_TASK_ROOT}"]

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
# CMD [ "lambda_function.lambda_handler" ]
CMD [ "lambda_function_parallel.lambda_handler" ]