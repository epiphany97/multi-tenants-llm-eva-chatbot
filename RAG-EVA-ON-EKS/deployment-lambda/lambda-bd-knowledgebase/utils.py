import json

def read_qa(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['question'], data['answer']

def read_llm_options(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def write_generated_results(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f)
    