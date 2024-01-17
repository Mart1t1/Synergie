import requests

url = "http://localhost:8000/evalsession"

filepath = "ressources/sample.csv"

files = {"file": open(filepath, "rb")}

r = requests.get(url, files = files)

print(r.content)


