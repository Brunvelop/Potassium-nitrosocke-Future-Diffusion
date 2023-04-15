import requests

url = "http://localhost:8000/"
headers = {"Content-Type": "application/json"}
data = {"prompt": "A robot cat"}

response = requests.post(url, json=data, headers=headers)

print(response.text)