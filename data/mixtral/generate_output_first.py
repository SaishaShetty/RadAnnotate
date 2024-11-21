
from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import json

tokenizer = MistralTokenizer.from_file("/home/spshetty/RadAnnotate/data/mixtral/mixtral-model/tokenizer.model.v3")
model = Transformer.from_folder('/home/spshetty/RadAnnotate/data/mixtral/mixtral-model')

prompt = """
    <s><INST>
    [System]
   You are a radiological expert capable of:
            1. Generating annotations based on individual tokens, each labeled as one of:
                - ANAT-DP: Anatomy - Definitely Present
                - ANAT-DA: Anatomy - Definitely Absent
                - OBS-DP: Observation - Definitely Present
                - OBS-DA: Observation - Definitely Absent
            2. Using these annotations to generate realistic, diverse synthetic clinical reports related to chest radiology.

            Your task is to generate variations in findings, conditions, and anatomical structures to help build a comprehensive dataset. Ensure the annotations are accurate and medically plausible, and the reports use the annotations naturally.[/System]
    [User]
            Generate 5 synthetic clinical reports related to the chest that can be helpful to find different diverse data for our model. 
            Use single-token annotations for the following categories:
            - ANAT-DP: Anatomy - Definitely Present
            - ANAT-DA: Anatomy - Definitely Absent
            - OBS-DP: Observation - Definitely Present
            - OBS-DA: Observation - Definitely Absent

        One example of a report and its annotations is:
        '''
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
            },
        "Report": "Patient has been extubated . Lungs are clear . Normal cardiomediastinal and hilar silhouettes and pleural surfaces ."
        }'''[/User]

        [/System]Make sure to only respond with valid JSON.[/INST]
"""
completion_request = ChatCompletionRequest(messages=[UserMessage(content = prompt)])

tokens = tokenizer.encode_chat_completion(completion_request).tokens

out_tokens, _ = generate([tokens], model, max_tokens=6000, temperature=0.9, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

#print(result)
with open('outputs/generated_output_first_reports.json', 'w') as f:
    json.dump(result, f)
