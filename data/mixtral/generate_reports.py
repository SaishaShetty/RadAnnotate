
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
    You are a radiological expert capable of generating diverse synthetic clinical reports and their annotations in a structured JSON response. 
    Your task is to create variations in findings, conditions, and anatomical structures related to the chest to help build a comprehensive dataset.[/System]
    [User]
    The annotations can be one of the two entities : Observation and Anatomy. These labels can have two uncertainty levels : Definitely Present and Definitely Absent.
        1. ANAT-DA (Anatomy - Definitely Absent)
        2. ANAT-DP (Anatomy - Definitely Present)
        3. OBS-DA (Observation - Definitely Absent)
        4. OBS-DP (Observation - Definitely Present)
    Annotate only specific **single words** related to anatomy or observations. Do not annotate phrases or entire sentences. Focus only on individual, relevant terms.
    Generate 1 synthetic clinical reports related to the chest that can be helpful to find different diverse data for our model. 
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
        }'''[/User]
        [/System]Generate synthetic clinical reports in the similar JSON format. Ensure that the report and the annotations make sense. [/INST]
"""
completion_request = ChatCompletionRequest(messages=[UserMessage(content = prompt)])

tokens = tokenizer.encode_chat_completion(completion_request).tokens

out_tokens, _ = generate([tokens], model, max_tokens=6000, temperature=0.4, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])


response = json.loads(result)
print(json.dumps(response, indent=2))
