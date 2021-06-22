import requests

predict_comments = [
    'good',
    'bad',
    'awful',
    'idiot',
]

predict_request = [
    {
        "comment_text": c,
    }
    for c in predict_comments 
]
response = requests.post('http://localhost:5000/predict', json=predict_request)
response.raise_for_status()

print('reponse time (ms):', response.elapsed.total_seconds() * 1000)
print(response.json())

print('='*50)
print((' '*5) + 'test was successful!')
print('='*50)
