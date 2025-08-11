import requests

url = "http://127.0.0.1:5000/predict"

payload = {
    "challengeId": 0,
    "missionId": 1,
    "userId": "test_user",
    "tempImageId": "temp123",
    "imageUrl": "https://drive.google.com/uc?export=download&id=1tRRBELNLU0VquAw4UYbJPgVsj2HD3Skf"
}


headers = {
    "Content-Type": "application/json"
}

response = requests.post(url, json=payload, headers=headers)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())
