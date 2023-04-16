import requests
import base64
import io
from PIL import Image

def base64_to_jpeg(base64_str, output_file):
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    img.save(output_file, 'JPEG')

url = "http://localhost:8000/"
headers = {"Content-Type": "application/json"}
data = {"prompt": "A robot cat"}

response = requests.post(url, json=data, headers=headers)

print(response.text)