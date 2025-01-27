from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import json
import os
tokenizer = MistralTokenizer.from_file("mixtral-model/tokenizer.model.v3")
model = Transformer.from_folder('/home/spshetty/RadAnnotate/data_generation/mixtral/mixtral-model')

def mixtral_generate_data(total_samples=1, batch_size=1, output_file="outputs/relation_testing.json"):
    
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
                    - ANAT-DP: Anatomy - Definitely Present (e.g., "heart," "aorta").
                    - ANAT-DA: Anatomy - Definitely Absent (e.g., "no pericardium").
                    - OBS-DP: Observation - Definitely Present (e.g., "effusion").
                    - OBS-DA: Observation - Definitely Absent (e.g., "no edema").
                2. Using these annotations to generate realistic, diverse synthetic clinical reports related to chest radiology. Annotations should be single words and not phrases.

                Instructions:
                1. **Generate a List of Tokens:**
                    - Create between 20 and 30 tokens per report.
                    - Prioritize generating more OBS-DA tokens.
                    - Include at least 3 different anatomical structures and 3 different observations.
                    - Make sure to not assume adjectives as a part of the anatomy. (Example - "pulmonary" is not an anatomy)
                    - Make sure not to annotate auxiliary words or modifiers as entities. (Example - in the sentence "No pleural effusion," the word "No" is not an annotation. Instead, only "effusion" should be labeled as OBS-DA. Apply this principle to similar cases. )

                2. **Construct a Report:**
                    - Make sure to use only the tokens provided in the annotations to construct the report. Do not add, modify, or interpret tokens beyond what is defined.
                    - Ensure every annotation is included in the report without omission, and all tokens align with their label definitions.
                    - Strictly avoid generating or using any tokens not provided in the annotations.

                3. **Incorporate Relations into the Report:**
                    - **suggestive_of (Observation, Observation):** Create pairs of Observation entities where the second logically depends on the first.
                        - Example: "scoliosis" is suggestive_of "asymmetry".
                            - Report: "Moderate scoliosis, causing asymmetry of the ribcage."

                    - **located_at (Observation, Anatomy):** Link Observation entities to appropriate Anatomy entities indicating their location or relationship.
                        - Example: "Normal" is located_at "cardiomediastinal".
                            - Report: "The cardiomediastinal contours appear normal."

                    - **modify (Observation, Observation) or (Anatomy, Anatomy):** Ensure one entity modifies or quantifies the degree of the second.
                        - Example 1: "Endotracheal" modifies "tube".
                            - Report: "The endotracheal tube tip terminates approximately 4.6 cm from the carina." (Observation, Observation)

                        - Example 2: "right" modifies "apex".
                            - Report: "The right apex is slightly obscured." (Anatomy, Anatomy)

                    **Note**An annotation may be linked to one or more annotations through defined relations, but this is not mandatory.

                4. **Output Format:**
                    Output strictly in JSON format as a list of dictionaries, with each dictionary having two keys:
                        - **`Report`:** The textual report.
                        - **`Annotations`:** A dictionary of token-level annotations and their relations, sorted by `start_ix`.

                5. **Validate the Output:**
                    - After generating the tokens and report, validate:
                        - That all relations reference valid tokens.
                        - That tokens align with the report text and their labels.
                        - Make sure every token in the list is used in the report.
                    - Ensure logical relationships (e.g., `effusion located_at lungs`) and flag/report illogical outputs (e.g., `effusion located_at trachea`).
                    
                6. **Realism in the Report:**
                    - Ensure every observation token has a logical anatomical location.
                    - Avoid creating reports with illogical relationships or ambiguous contexts.

                [User]
                Generate synthetic clinical reports related to the chest that can be helpful to find different diverse data for our model.
                Use single-token annotations for the following categories:
                    - ANAT-DP: Anatomy - Definitely Present
                    - ANAT-DA: Anatomy - Definitely Absent
                    - OBS-DP: Observation - Definitely Present
                    - OBS-DA: Observation - Definitely Absent

                Two examples of a report and its annotations is:
                {
                "Annotations": {
                        "1": {
                            "tokens": "left",
                            "label": "ANAT-DP",
                            "start_ix": 48,
                            "end_ix": 48,
                            "relations": [
                            [
                                "modify",
                                "2"
                            ]
                            ]
                        },
                        "2": {
                            "tokens": "basal",
                            "label": "ANAT-DP",
                            "start_ix": 49,
                            "end_ix": 49,
                            "relations": []
                        },
                        "3": {
                            "tokens": "parenchymal",
                            "label": "ANAT-DP",
                            "start_ix": 50,
                            "end_ix": 50,
                            "relations": [
                            [
                                "modify",
                                "2"
                            ]
                            ]
                        },
                        "4": {
                            "tokens": "opacity",
                            "label": "OBS-DA",
                            "start_ix": 51,
                            "end_ix": 51,
                            "relations": [
                            [
                                "located_at",
                                "3"
                            ]
                            ]
                        },
                        "5": {
                            "tokens": "remnant",
                            "label": "OBS-DA",
                            "start_ix": 59,
                            "end_ix": 59,
                            "relations": [
                            [
                                "modify",
                                "6"
                            ]
                            ]
                        },
                        "6": {
                            "tokens": "opacities",
                            "label": "OBS-DA",
                            "start_ix": 60,
                            "end_ix": 60,
                            "relations": []
                        },
                        "7": {
                            "tokens": "complications",
                            "label": "OBS-DA",
                            "start_ix": 63,
                            "end_ix": 63,
                            "relations": []
                        },
                        "8": {
                            "tokens": "frontal",
                            "label": "OBS-DP",
                            "start_ix": 67,
                            "end_ix": 67,
                            "relations": [
                            [
                                "modify",
                                "10"
                            ]
                            ]
                        },
                        "9": {
                            "tokens": "lateral",
                            "label": "OBS-DP",
                            "start_ix": 70,
                            "end_ix": 70,
                            "relations": [
                            [
                                "modify",
                                "10"
                            ]
                            ]
                        },
                        "10": {
                            "tokens": "radiograph",
                            "label": "OBS-DP",
                            "start_ix": 71,
                            "end_ix": 71,
                            "relations": []
                        },
                        "11": {
                            "tokens": "unremarkable",
                            "label": "OBS-DP",
                            "start_ix": 73,
                            "end_ix": 73,
                            "relations": [
                            [
                                "modify",
                                "10"
                            ]
                            ]
                        },
                        "12": {
                            "tokens": "pleural",
                            "label": "ANAT-DP",
                            "start_ix": 76,
                            "end_ix": 76,
                            "relations": []
                        },
                        "13": {
                            "tokens": "effusions",
                            "label": "OBS-DA",
                            "start_ix": 77,
                            "end_ix": 77,
                            "relations": [
                            [
                                "located_at",
                                "12"
                            ]
                            ]
                        },
                        "14": {
                            "tokens": "pulmonary",
                            "label": "ANAT-DP",
                            "start_ix": 80,
                            "end_ix": 80,
                            "relations": []
                        },
                        "15": {
                            "tokens": "edema",
                            "label": "OBS-DA",
                            "start_ix": 81,
                            "end_ix": 81,
                            "relations": [
                            [
                                "located_at",
                                "14"
                            ]
                            ]
                        },
                        "16": {
                            "tokens": "Moderate",
                            "label": "OBS-DP",
                            "start_ix": 83,
                            "end_ix": 83,
                            "relations": [
                            [
                                "modify",
                                "17"
                            ]
                            ]
                        },
                        "17": {
                            "tokens": "scoliosis",
                            "label": "OBS-DP",
                            "start_ix": 84,
                            "end_ix": 84,
                            "relations": [
                            [
                                "suggestive_of",
                                "18"
                            ]
                            ]
                        },
                        "18": {
                            "tokens": "asymmetry",
                            "label": "OBS-DP",
                            "start_ix": 87,
                            "end_ix": 87,
                            "relations": [
                            [
                                "located_at",
                                "19"
                            ]
                            ]
                        },
                        "19": {
                            "tokens": "ribcage",
                            "label": "ANAT-DP",
                            "start_ix": 90,
                            "end_ix": 90,
                            "relations": []
                        }
                    "Report": "As compared to the previous radiograph , a pre - existing left basal parenchymal opacity has completely cleared . No evidence of remnant opacities or of complications . Both the frontal and the lateral radiograph appear unremarkable . No pleural effusions . No pulmonary edema . Moderate scoliosis , causing asymmetry of the ribcage ."
                    },
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
                        }
                    } 
                "Report": "Patient has been extubated . Lungs are clear . Normal cardiomediastinal and hilar silhouettes and pleural surfaces ."
                },[/User]

                [/System]
                **Important** Your output must strictly start with `{` and end with `}`. In batches you can start with '[' but do not end with ']'. Do not include any text, explanations, or descriptions outside the JSON structure. If you fail to comply, the output will be considered invalid. After every report generated add a ','[/INST]
                """
                
        completion_request = ChatCompletionRequest(messages=[UserMessage(content = prompt)])

        tokens = tokenizer.encode_chat_completion(completion_request).tokens

        out_tokens, _ = generate([tokens], model, max_tokens=32000, temperature=0.6, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])
        #parsed_res = json.loads(result) 
        print(result)
        parsed_res = json.loads(result)  
        print(parsed_res)

        with open(output_file, "a") as file:
            json.dump(parsed_res, file, indent=4)  # Saves the entire parsed_res instead of the current obj
            file.write(",\n")

        total_results += len(parsed_res)  # Update total results counter"""

    print(f"Generated {total_results} reports and saved to {output_file}")


mixtral_generate_data(total_samples=1, batch_size=1, output_file="outputs/try.json")

##40,10
