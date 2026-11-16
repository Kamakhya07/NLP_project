import urllib.request
import json

url = 'http://127.0.0.1:5000/api/predict'
payloads = [
    'i like your new reel!',
    'you are an ugly loser go die',
    'great editing and transitions',
    'shut up nobody asked for this stupid content'
]

print("="*60)
print("TESTING /api/predict ENDPOINT")
print("="*60)
for text in payloads:
    data = json.dumps({'text': text}).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode())
        print(f"{res['prediction']:12s} | should_hide={str(res['should_hide']):5s} ({res['confidence']:.1%}) | {text}")
print("="*60)
