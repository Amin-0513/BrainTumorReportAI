import base64
import requests

with open(
    r"E:\Brain tumor Detection\LLMServices\lime_explanation_output12.png",
    "rb"
) as img:
    image_base64 = base64.b64encode(img.read()).decode("utf-8")

payload = {
    "image_base64": image_base64,
    "prediction": "glioma"
}

response = requests.post(
    "http://127.0.0.1:5000//generate-report",
    json=payload
)

print(response.json())
