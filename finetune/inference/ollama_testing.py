import requests
import json

report_text = 'The lungs are clear . Cardiomediastinal and hilar contours are normal . There are no pleural effusions or pneumothorax .'
prompt = f"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert radiology assistant that helps annotate reports<|eot_id|><|start_header_id|>user<|end_header_id|>
Follow these guidelines:\n
1. Label terms as anatomy if they clearly refer to a body part and, if present, include any modifier and descriptors.\n
2. Label terms as observation if they describe a state, condition, or finding related to the anatomy.\n
Do not label:\n
    1. Medical devices\n
    2. Medical conditions or descriptive terms related to conditions (usually an observation)\n
    3. Adjectives describing an anatomy part\n

Label each word seperately. Go through every word.
    Report: {report_text}
<|eot_id|><|start_header_id|>user<|end_header_id|>
 Provide the annotated list with proper reasoning for each: <|start_header_id|>Assistant<|end_header_id|>
   
<|end_of_text|>
    """

url = "http://localhost:11434/api/generate"
headers = {"Content-Type": "application/json"}
data = {
    "model": "llama3.1",
    "prompt": prompt,
    "temperature": 0,
    "top_p": 0,
    "top_k": 1,
    "stream" : False
}

response = requests.post(url, headers=headers, data=json.dumps(data))
result = response.json()

if "response" in result:
    print(result["response"])
else:
    print("No response received")