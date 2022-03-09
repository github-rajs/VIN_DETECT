
from pydoc import resolve
import requests


BASE='http://127.0.0.1:5000/'

response=requests.get(BASE + "test/",stream=False,timeout=30)
print(response.json())

#input()
#response=requests.get(BASE + "test/4")
#print(response.json()) 