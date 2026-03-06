import json
import urllib.request
req = urllib.request.Request('http://localhost:8000/query', data=json.dumps({'question': 'Which product categories generate the most revenue but also experience the highest late delivery risk?'}).encode('utf-8'), headers={'Content-Type': 'application/json'})
response = urllib.request.urlopen(req)
result = json.loads(response.read().decode('utf-8'))
print(result['sql'])
