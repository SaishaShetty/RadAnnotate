import torch
from transformers import pipeline
import json
model_id = "meta-llama/Llama-3.2-3B-Instruct"

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {
        "role": "system",
        "content": (
            """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a radiological expert capable of generating diverse synthetic clinical reports and their annotations. 
            Your task is to create variations in findings, conditions, and anatomical structures related to the chest to help build a comprehensive dataset."""
        )
    },
    {"role": "user", 
    "content": (
        """
        Generate 5 synthetic clinical reports related to the chest that can be helpful to find different diverse data for our model. 
        One example of a report and its annotations is : 
        '''
        "Report": "Patient has been extubated . Lungs are clear . Normal cardiomediastinal and hilar silhouettes and pleural surfaces .",
        "Annotations": {
            "1": {
                "tokens": "Lungs",
                "label": "ANAT-DP",
                "start_ix": 36,
                "end_ix": 36,
                "relations": []
            },
            "2": {
                "tokens": "clear",
                "label": "OBS-DP",
                "start_ix": 38,
                "end_ix": 38,
                "relations": [
                    [
                        "located_at",
                        "1"
                    ]
                ]
            },
            "3": {
                "tokens": "Normal",
                "label": "OBS-DP",
                "start_ix": 40,
                "end_ix": 40,
                "relations": [
                    [
                        "located_at",
                        "4"
                    ],
                    [
                        "located_at",
                        "5"
                    ],
                    [
                        "located_at",
                        "7"
                    ]
                ]
            },
            "4": {
                "tokens": "cardiomediastinal",
                "label": "ANAT-DP",
                "start_ix": 41,
                "end_ix": 41,
                "relations": []
            },
            "5": {
                "tokens": "hilar",
                "label": "ANAT-DP",
                "start_ix": 43,
                "end_ix": 43,
                "relations": []
            },
            "6": {
                "tokens": "silhouettes",
                "label": "ANAT-DP",
                "start_ix": 44,
                "end_ix": 44,
                "relations": [
                    [
                        "modify",
                        "4"
                    ],
                    [
                        "modify",
                        "5"
                    ]
                ]
            },
            "7": {
                "tokens": "pleural",
                "label": "ANAT-DP",
                "start_ix": 46,
                "end_ix": 46,
                "relations": []
            },
            "8": {
                "tokens": "surfaces",
                "label": "ANAT-DP",
                "start_ix": 47,
                "end_ix": 47,
                "relations": [
                    [
                        "modify",
                        "7"
                    ]
                ]
            }
        }'''
            Generate in the similar JSON format. No additional text should be included."""
            )
    }
    
]
outputs = pipe(
    messages,
    max_new_tokens=6000,
    temperature = 0.9,
    top_p = 0.95,
    top_k = 50
)
print(outputs[0]["generated_text"])
