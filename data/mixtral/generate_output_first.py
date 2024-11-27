from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import json
import os
tokenizer = MistralTokenizer.from_file("/home/spshetty/RadAnnotate/data/mixtral/mixtral-model/tokenizer.model.v3")
model = Transformer.from_folder('/home/spshetty/RadAnnotate/data/mixtral/mixtral-model')

def mixtral_generate_data(total_samples=100, batch_size=10, output_file="outputs/data_100.json"):
    
    if not os.path.exists(output_file):
        with open(output_file, "w") as file:
            pass  

    total_results = 0  

    for batch_start in range(0, total_samples, batch_size):
        remaining_samples = min(batch_size, total_samples - batch_start)
        prompt = """
            <s><INST>
            [System]
                You are a radiological expert capable of:
                    1. Generating annotations based on individual tokens, each labeled as one of:
                        - ANAT-DP: Anatomy - Definitely Present
                        - ANAT-DA: Anatomy - Definitely Absent
                        - OBS-DP: Observation - Definitely Present
                        - OBS-DA: Observation - Definitely Absent
                    2. Using these annotations to generate realistic, diverse synthetic clinical reports related to chest radiology. Annotations should be single words and not phrases.

                    Instructions:
                    - First, generate a list of tokens (annotations), ensuring diversity across anatomy and observations.
                    - Then, use these tokens to construct a report. Ensure every token appears naturally in the report.
                    - Avoid repeatedly starting reports with the same token, such as 'Lungs.' Ensure diversity in the choice of initial annotations across reports while maintaining medical plausibility.
                    - Make sure to number every token in every report.
                    -  Stricly output in JSON format as a list of dictionaries. Each dictionary should have two keys 'Report' and 'Annotations'. [/System]
                [User]

                    Generate 10 synthetic clinical reports related to the chest that can be helpful to find different diverse data for our model. 
                    Use single-token annotations for the following categories:
                    - ANAT-DP: Anatomy - Definitely Present
                    - ANAT-DA: Anatomy - Definitely Absent
                    - OBS-DP: Observation - Definitely Present
                    - OBS-DA: Observation - Definitely Absent

                One example of a report and its annotations is:
                {
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
                },[/User]

                [/System]
                **Important** Strictly just output the JSON.[/INST]
        """
        completion_request = ChatCompletionRequest(messages=[UserMessage(content = prompt)])

        tokens = tokenizer.encode_chat_completion(completion_request).tokens

        out_tokens, _ = generate([tokens], model, max_tokens=20000, temperature=0.6, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        print(result)
        parsed_res = json.loads(result)  
        print(parsed_res)
        
        with open(output_file, "a") as file:
            for obj in parsed_res:
                json.dump(parsed_res, file, indent=4)  # Pretty-print each JSON object
                file.write(",\n") 

        total_results += len(parsed_res)  # Update total results counter

    print(f"Generated {total_results} reports and saved to {output_file}")


mixtral_generate_data(total_samples=100, batch_size=10, output_file="outputs/data_100.json")

    
