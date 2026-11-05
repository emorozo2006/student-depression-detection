import requests
url='https://falling-bird-831.fly.dev/predict'

student_data={
  "gender": "male",
  "family_background": "middle_class",
  "relationship_status": "complicated",
  "marital_status": "unmarried",
  "living_style": "family",
  "friend_circle": "no",
  "home_division": "dhaka",
  "current_semester": 2,
  "age": 20,
  "cgpa": 3.77,
  "cost_without_study": 17413
}

response = requests.post(url, json=student_data).json()
print("Prediction response:", response)