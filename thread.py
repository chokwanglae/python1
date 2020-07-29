import json
from urllib.request import urlopen
url = "https://raw.githubusercontent.com/AstinCHOI/introducing-python/master/intro/top_rated.json"
response =urlopen(url)
contents=response.read()
text=contents.decode('utf8')
data =json.loads(text)
for video in data['feed']['entry'][0:8]:
	print(video['title']['$t'])