from pydoc import resolve
import requests

BASE='http://127.0.0.1:5002/API_EXAMPLE/'
image='fake1.jpg'
response=requests.get(BASE + image)
print(response.json())

#input()
#response=requests.get(BASE + "test/4")
#print(response.json()) 