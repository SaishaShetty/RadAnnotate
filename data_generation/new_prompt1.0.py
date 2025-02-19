from mistral_inference.transformer import Transformer
from mistral_inference.generate import generate
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
import json
import os
import torch

tokenizer = MistralTokenizer.from_file("/home/spshetty/RadAnnotate/data_generation/mixtral/mixtral-model/tokenizer.model.v3")

def mixtral_generate_data(total_samples=1, batch_size=1, output_file="/home/spshetty/RadAnnotate/data_generation/outputs/check.json"):
    global model
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as file:
        file.write("[\n")

    total_results = 0  

    for batch_start in range(0, total_samples, batch_size):
        torch.cuda.empty_cache()  # Clear GPU memory before every run

        # Reload model to avoid cache issues
        if "model" in globals():
            del model  # Delete model if it exists to avoid memory issues

        model = Transformer.from_folder('/home/spshetty/RadAnnotate/data_generation/mixtral/mixtral-model')
        
        prompt = """
            <s><INST>
            [System]
                You are a radiological expert capable of:
                1. Generating annotations based on individual tokens, each labeled as one of:
                    - ANAT-DP: Anatomy - Definitely Present (e.g., "heart," "aorta").
                    - ANAT-DA: Anatomy - Definitely Absent (e.g., "no pericardium").
                    - OBS-DP: Observation - Definitely Present (e.g., "effusion").
                    - OBS-DA: Observation - Definitely Absent (e.g., "no edema").
                2. Using these annotations to generate realistic, diverse synthetic clinical reports related to chest radiology.


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
                },
                {
                "Annotations": {
                    "1": {
                        "tokens": "Lung",
                        "label": "ANAT-DP",
                        "start_ix": 39,
                        "end_ix": 39,
                        "relations": []
                    },
                    "2": {
                        "tokens": "volumes",
                        "label": "ANAT-DP",
                        "start_ix": 40,
                        "end_ix": 40,
                        "relations": [
                        [
                            "modify",
                            "1"
                        ]
                        ]
                    },
                    "3": {
                        "tokens": "low",
                        "label": "OBS-DP",
                        "start_ix": 42,
                        "end_ix": 42,
                        "relations": [
                        [
                            "located_at",
                            "1"
                        ]
                        ]
                    },
                    "4": {
                        "tokens": "pneumonia",
                        "label": "OBS-DA",
                        "start_ix": 52,
                        "end_ix": 52,
                        "relations": []
                    },
                    "5": {
                        "tokens": "overt",
                        "label": "OBS-DA",
                        "start_ix": 54,
                        "end_ix": 54,
                        "relations": [
                        [
                            "modify",
                            "6"
                        ]
                        ]
                    },
                    "6": {
                        "tokens": "CHF",
                        "label": "OBS-DA",
                        "start_ix": 55,
                        "end_ix": 55,
                        "relations": []
                    },
                    "7": {
                        "tokens": "effusion",
                        "label": "OBS-DA",
                        "start_ix": 61,
                        "end_ix": 61,
                        "relations": []
                    },
                    "8": {
                        "tokens": "pneumothorax",
                        "label": "OBS-DA",
                        "start_ix": 63,
                        "end_ix": 63,
                        "relations": []
                    },
                    "9": {
                        "tokens": "cardiomediastinal",
                        "label": "ANAT-DP",
                        "start_ix": 66,
                        "end_ix": 66,
                        "relations": []
                    },
                    "10": {
                        "tokens": "silhouette",
                        "label": "ANAT-DP",
                        "start_ix": 67,
                        "end_ix": 67,
                        "relations": [
                        [
                            "modify",
                            "9"
                        ]
                        ]
                    },
                    "11": {
                        "tokens": "normal",
                        "label": "OBS-DP",
                        "start_ix": 69,
                        "end_ix": 69,
                        "relations": [
                        [
                            "located_at",
                            "9"
                        ]
                        ]
                    },
                    "12": {
                        "tokens": "osseous",
                        "label": "ANAT-DP",
                        "start_ix": 72,
                        "end_ix": 72,
                        "relations": []
                    },
                    "13": {
                        "tokens": "structures",
                        "label": "ANAT-DP",
                        "start_ix": 73,
                        "end_ix": 73,
                        "relations": [
                        [
                            "modify",
                            "12"
                        ]
                        ]
                    },
                    "14": {
                        "tokens": "intact",
                        "label": "OBS-DP",
                        "start_ix": 75,
                        "end_ix": 75,
                        "relations": [
                        [
                            "located_at",
                            "12"
                        ]
                        ]
                    },
                    "15": {
                        "tokens": "Limited",
                        "label": "OBS-DP",
                        "start_ix": 79,
                        "end_ix": 79,
                        "relations": []
                    },
                    "16": {
                        "tokens": "negative",
                        "label": "OBS-DP",
                        "start_ix": 81,
                        "end_ix": 81,
                        "relations": []
                    }
                "Report": "Lung volumes are low . Allowing for this , no definite evidence of pneumonia or overt CHF . No supine evidence for effusion or pneumothorax . The cardiomediastinal silhouette is normal . Imaged osseous structures are intact "
                },
                "Annotations":{
                    "1": {
                        "tokens": "Borderline",
                        "label": "OBS-DP",
                        "start_ix": 34,
                        "end_ix": 34,
                        "relations": [
                        [
                            "located_at",
                            "3"
                        ]
                        ]
                    },
                    "2": {
                        "tokens": "size",
                        "label": "ANAT-DP",
                        "start_ix": 35,
                        "end_ix": 35,
                        "relations": [
                        [
                            "modify",
                            "3"
                        ]
                        ]
                    },
                    "3": {
                        "tokens": "cardiac",
                        "label": "ANAT-DP",
                        "start_ix": 38,
                        "end_ix": 38,
                        "relations": []
                    },
                    "4": {
                        "tokens": "silhouette",
                        "label": "ANAT-DP",
                        "start_ix": 39,
                        "end_ix": 39,
                        "relations": [
                        [
                            "modify",
                            "3"
                        ]
                        ]
                    },
                    "5": {
                        "tokens": "pneumonia",
                        "label": "OBS-DA",
                        "start_ix": 42,
                        "end_ix": 42,
                        "relations": []
                    },
                    "6": {
                        "tokens": "pulmonary",
                        "label": "ANAT-DP",
                        "start_ix": 45,
                        "end_ix": 45,
                        "relations": []
                    },
                    "7": {
                        "tokens": "edema",
                        "label": "OBS-DA",
                        "start_ix": 46,
                        "end_ix": 46,
                        "relations": [
                        [
                            "located_at",
                            "6"
                        ]
                        ]
                    },
                    "8": {
                        "tokens": "pneumothorax",
                        "label": "OBS-DA",
                        "start_ix": 53,
                        "end_ix": 53,
                        "relations": []
                    }
                "Report": "As compared to the previous radiograph , there is no relevant change . Borderline size of the cardiac silhouette . No pneumonia , no pulmonary edema. No pneumothorax."
                },
                [/User]

                Generate the synthetic annotations and report by following the instructions in order:
                1. **Generate a List of Tokens:**
                    - Create between 20 and 30 tokens per report.
                    - Include at least 3 different anatomical structures and 3 different observations.
                    - Make sure to not assume adjectives as a part of the anatomy. (Example - "pulmonary" is not an anatomy)
                    - Make sure not to annotate auxiliary words or modifiers as entities. (Example - in the sentence "No pleural effusion," the word "No" is not an annotation. Instead, only "effusion" should be labeled as OBS-DA. Apply this principle to similar cases. )
                    - Stick strictly to generating individual words and not phrases.
                    - Avoid starting multiple reports with the same token, such as 'Lungs.' Ensure diversity in the choice of initial annotations across reports.

                2. **Construct a Report:**
                    - Stricly construct the report using only the tokens provided in the annotations. Do not add, modify, or interpret tokens beyond what is defined.
                    - Ensure every annotation is included in the report without omission, and all tokens align with their label definitions.
                    - Make sure to not generate or use any tokens not provided in the annotations.

                3. **Incorporate Relations into the Report:**
                    1. **`suggestive_of (Observation → Observation):`** Link `Observation` entities where the second logically depends on or is inferred from the first.  
                        - **Example:** In the sentence "Moderate scoliosis, causing asymmetry of the ribcage.", "scoliosis" is suggestive_of "asymmetry".

                    2. **`located_at (Observation → Anatomy):`**  Link `Observation` entities to the corresponding `Anatomy` entities to indicate location or spatial relationship.  
                        - **Example:**  In the sentence "The cardiomediastinal contours appear normal.", "Normal" is located_at "cardiomediastinal".
                        
                    3. **`modify (Observation → Observation)` or `(Anatomy → Anatomy):`**  Annotate when one entity modifies, specifies, or quantifies the degree of another entity.  
                        - **Example 1 (**Observation → Observation**):**  In the sentence "The endotracheal tube tip terminates approximately 4.6 cm from the carina.",  "Endotracheal" is modify at "tube".
                        - **Example 2 (Anatomy → Anatomy): In the sentence "The left apex is slightly obscured.", "left" is modify at "apex". 

                    *Important* Ensure directional terms (eg. left, right,etc) in a report should have a "modify" relation with the corresponding anatomy.

                    **Note**An annotation may be linked to one or more annotations through defined relations, but this is not mandatory.

                4. **Output Format:**
                    Output strictly in JSON format as a list of dictionaries, with each dictionary having two keys:
                        - **`Annotations`:** A dictionary of token-level annotations and their relations, sorted by `start_ix`.
                        - **`Report`:** The textual report.
                        
                5. **Validate the Output:**
                    - After generating the tokens and report, validate:
                        - That all relations reference valid tokens.
                        - That tokens align with the report text and their labels.
                        - Make sure every token in the list is used in the report.
                    - Ensure logical relationships (e.g., `effusion located_at lungs`) and flag/report illogical outputs (e.g., `effusion located_at trachea`).
                    
                6. **Realism in the Report:**
                    - Ensure every observation token has a logical anatomical location.
                    - Avoid creating reports with illogical relationships or ambiguous contexts.
                
                7. **Validation:**
                    - Ensure that all the annotations produced are used in generating the report. They should logically fit.
                
                Important Note : Generate the current reports using "heart, aorta, pulmonary, arteries,etc" these tokens.


                [/System]
                **Important** Your output must strictly start with `{` and end with `}` only. Do not include any text, explanations, or descriptions outside the JSON structure. If you fail to comply, the output will be considered invalid. After every report generated add a ','[/INST]
                """
                
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        tokens = tokenizer.encode_chat_completion(completion_request).tokens

        try:
            out_tokens, _ = generate(
                [tokens], 
                model, 
                max_tokens=32000,  
                temperature=0.6, 
                eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id
            )
        except RuntimeError as e:
            print(f"Runtime Error: {e}")
            torch.cuda.empty_cache()
            continue  

        result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

        try:
            parsed_res = json.loads(result)
        except json.JSONDecodeError as e:
            print(f"JSON Error: {e}")
            parsed_res = {"error": "Invalid JSON output"}

        with open(output_file, "a") as file:
            json.dump(parsed_res, file, indent=4)
            if batch_start + batch_size < total_samples:
                file.write(",\n")

        total_results += len(parsed_res) if isinstance(parsed_res, list) else 1

    with open(output_file, "a") as file:
        file.write("\n]")

    print(f"Generated {total_results} reports and saved to {output_file}")
mixtral_generate_data(total_samples=1, batch_size=1, output_file="/home/spshetty/RadAnnotate/data_generation/outputs/check.json")
