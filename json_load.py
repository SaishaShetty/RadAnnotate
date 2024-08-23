import json
from utils import llama
prompt = "Write a short quote to motivate me"
response = llama(prompt)
"""f = open('data/train.json')
data = json.load(f)
for x in data:
    for i in x:
        print(i)
   """