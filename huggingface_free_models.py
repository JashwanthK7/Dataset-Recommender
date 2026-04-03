import requests

def find_free_models():
    url = 'https://huggingface.co/api/models'
    params = {
        'pipeline_tag': 'text-generation',
        'inference': 'warm'
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    if isinstance(data, dict):
        print(data)
    elif isinstance(data, list):
        for model in data[:10]:
            print(model.get('id'))

find_free_models()