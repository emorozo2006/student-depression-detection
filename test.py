import requests

url = 'http://localhost:9696/predict'

student_data = {
    "gender": "male",
    "age": 22,
    "cgpa": 3.4,
    "family_background": "middle_class",
    "relationship_status": "single",
    "marital_status": "unmarried",
    "living_style": "hostel",
    "current_semester": "6",
    "friend_circle": "yes",
    "home_division": "dhaka",
    "cost_without_study": 12000
}

response = requests.post(url, json=student_data)

print(response.json())