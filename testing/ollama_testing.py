import requests
import json

report_text = "Evaluate for effusions . There is mild - to - moderate cardiomegaly . Bilateral pleural effusions are small . Aside from atelectasis in the left lower lobe , the lungs are grossly clear . Almost complete resolution of atelectasis in the left upper lobe . Sternal wires are aligned . Widened mediastinum has improved . A small air - fluid level in the retrosternal region suggests the presence of a tiny pneumothorax and small effusion . These are most likely located in the left side . "
prompt = f"""
<|begin_of_text|><|start_header_id|>Assistant<|end_header_id|>
     Your task is to extract anatomy terms exactly as they appear in the report. Do not modify, infer, or shorten terms in any way. Only output terms that appear word-for-word in the text. Go through each individual word.

    [THOUGHT_START]
    The terms should solely be an anatomy in the body. **Provide proper reasoning for why you labelled each of them as an anatomy**. 
    If your reason does not indicate it is a anatomy part only then you should not label it. If the anatomy term has a modifier include those as well. 
    MAKE SURE TO NOT label these terms as anatomy:
    1. Medical device
    2. Medical condition or description of a condition
    3. An adjactive describing an anatomy part
    
    If a label includes two words, separate them but still keep them as labels.
    Once that is done, give me the final list of terms that exactly appear in the report.
    [THOUGHT_END]

    Report: {report_text}

    Output:
    [ANATOMY] term1, term2  
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