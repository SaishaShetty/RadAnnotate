import subprocess
import json
format = """{
{
   
        "text": "FINAL REPORT EXAMINATION : CHEST ( PORTABLE AP ) INDICATION : ___ year old woman with SAH / / Fever workup Fever workup IMPRESSION : Compared to chest radiographs ___ . Patient has been extubated . Lungs are clear . Normal cardiomediastinal and hilar silhouettes and pleural surfaces .",
        
        "entities": {
            "1": {
                "tokens": "Patient",
                "label": "ANAT-DP",
                "start_ix": 31,
                "end_ix": 31,
                "relations": []
            },
            "2": {
                "tokens": "extubated",
                "label": "CHAN-DEV-DISA",
                "start_ix": 34,
                "end_ix": 34,
                "relations": [
                    [
                        "modify",
                        "1"
                    ]
                ]
            },
            "3": {
                "tokens": "Lungs",
                "label": "ANAT-DP",
                "start_ix": 36,
                "end_ix": 36,
                "relations": []
            },
            "4": {
                "tokens": "clear",
                "label": "OBS-DP",
                "start_ix": 38,
                "end_ix": 38,
                "relations": [
                    [
                        "located_at",
                        "3"
                    ]
                ]
            },
            "5": {
                "tokens": "Normal",
                "label": "OBS-DP",
                "start_ix": 40,
                "end_ix": 40,
                "relations": [
                    [
                        "located_at",
                        "6"
                    ],
                    [
                        "located_at",
                        "7"
                    ],
                    [
                        "located_at",
                        "9"
                    ]
                ]
            },
            "6": {
                "tokens": "cardiomediastinal",
                "label": "ANAT-DP",
                "start_ix": 41,
                "end_ix": 41,
                "relations": []
            },
            "7": {
                "tokens": "hilar",
                "label": "ANAT-DP",
                "start_ix": 43,
                "end_ix": 43,
                "relations": []
            },
            "8": {
                "tokens": "silhouettes",
                "label": "ANAT-DP",
                "start_ix": 44,
                "end_ix": 44,
                "relations": [
                    [
                        "modify",
                        "6"
                    ],
                    [
                        "modify",
                        "7"
                    ]
                ]
            },
            "9": {
                "tokens": "pleural",
                "label": "ANAT-DP",
                "start_ix": 46,
                "end_ix": 46,
                "relations": []
            },
            "10": {
                "tokens": "surfaces",
                "label": "ANAT-DP",
                "start_ix": 47,
                "end_ix": 47,
                "relations": [
                    [
                        "modify",
                        "9"
                    ]
                ]
            }

    },
    {
        "text": "FINAL REPORT EXAMINATION : CHEST ( PORTABLE AP ) INDICATION : History : ___ F with ett placement TECHNIQUE : Upright AP view of the chest COMPARISON : None . Patient is currently listed as EU critical . FINDINGS : Endotracheal tube tip terminates approximately a 4.6 cm from the carina . Enteric tube tip terminates within the distal esophagus and should be advanced by at least 11 cm . Heart size is normal . The mediastinal and hilar contours are grossly unremarkable . Pulmonary vasculature is not engorged . Apart from minimal atelectasis in the lung bases , the lungs appear clear . No large pleural effusion or pneumothorax is present though the extreme right apex is slightly obscured by the patient 's chin projecting over this area . No displaced fractures are present . No acute osseous abnormality is seen . Gaseous distension of the stomach is noted . IMPRESSION : 1 . Standard positioning of the endotracheal tube . 2 . Enteric tube tip is suboptimally located within the distal esophagus and should be advanced by at least 11 cm . Gaseous distention of the stomach is also noted . 3 . No focal consolidation .",
        "entities": {
            "1": {
                "tokens": "Endotracheal",
                "label": "OBS-DP",
                "start_ix": 40,
                "end_ix": 40,
                "relations": [
                    [
                        "modify",
                        "2"
                    ]
                ]
            },
            "2": {
                "tokens": "tube",
                "label": "OBS-DP",
                "start_ix": 41,
                "end_ix": 41,
                "relations": []
            },
            "3": {
                "tokens": "tip",
                "label": "OBS-DP",
                "start_ix": 42,
                "end_ix": 42,
                "relations": [
                    [
                        "modify",
                        "2"
                    ],
                    [
                        "located_at",
                        "5"
                    ]
                ]
            },
            "4": {
                "tokens": "approximately a 4.6 cm",
                "label": "ANAT-DP",
                "start_ix": 44,
                "end_ix": 47,
                "relations": [
                    [
                        "modify",
                        "5"
                    ]
                ]
            },
            "5": {
                "tokens": "carina",
                "label": "ANAT-DP",
                "start_ix": 50,
                "end_ix": 50,
                "relations": []
            },
            "6": {
                "tokens": "Enteric",
                "label": "OBS-DP",
                "start_ix": 52,
                "end_ix": 52,
                "relations": [
                    [
                        "modify",
                        "7"
                    ]
                ]
            },
            "7": {
                "tokens": "tube",
                "label": "OBS-DP",
                "start_ix": 53,
                "end_ix": 53,
                "relations": []
            },
            "8": {
                "tokens": "tip",
                "label": "OBS-DP",
                "start_ix": 54,
                "end_ix": 54,
                "relations": [
                    [
                        "modify",
                        "7"
                    ],
                    [
                        "located_at",
                        "10"
                    ]
                ]
            },
            "9": {
                "tokens": "distal",
                "label": "ANAT-DP",
                "start_ix": 58,
                "end_ix": 58,
                "relations": [
                    [
                        "modify",
                        "10"
                    ]
                ]
            },
            "10": {
                "tokens": "esophagus",
                "label": "ANAT-DP",
                "start_ix": 59,
                "end_ix": 59,
                "relations": []
            },
            "11": {
                "tokens": "advanced by at least 11 cm",
                "label": "OBS-DP",
                "start_ix": 63,
                "end_ix": 68,
                "relations": [
                    [
                        "modify",
                        "8"
                    ]
                ]
            },
            "12": {
                "tokens": "Heart",
                "label": "ANAT-DP",
                "start_ix": 70,
                "end_ix": 70,
                "relations": []
            },
            "13": {
                "tokens": "size",
                "label": "ANAT-DP",
                "start_ix": 71,
                "end_ix": 71,
                "relations": [
                    [
                        "modify",
                        "12"
                    ]
                ]
            },
            "14": {
                "tokens": "normal",
                "label": "OBS-DP",
                "start_ix": 73,
                "end_ix": 73,
                "relations": [
                    [
                        "located_at",
                        "12"
                    ]
                ]
            },
            "15": {
                "tokens": "mediastinal",
                "label": "ANAT-DP",
                "start_ix": 76,
                "end_ix": 76,
                "relations": []
            },
            "16": {
                "tokens": "hilar",
                "label": "ANAT-DP",
                "start_ix": 78,
                "end_ix": 78,
                "relations": []
            },
            "17": {
                "tokens": "contours",
                "label": "ANAT-DP",
                "start_ix": 79,
                "end_ix": 79,
                "relations": [
                    [
                        "modify",
                        "15"
                    ],
                    [
                        "modify",
                        "16"
                    ]
                ]
            },
            "18": {
                "tokens": "grossly",
                "label": "OBS-DP",
                "start_ix": 81,
                "end_ix": 81,
                "relations": [
                    [
                        "modify",
                        "19"
                    ]
                ]
            },
            "19": {
                "tokens": "unremarkable",
                "label": "OBS-DP",
                "start_ix": 82,
                "end_ix": 82,
                "relations": [
                    [
                        "located_at",
                        "15"
                    ],
                    [
                        "located_at",
                        "16"
                    ]
                ]
            },
            "20": {
                "tokens": "Pulmonary",
                "label": "ANAT-DP",
                "start_ix": 84,
                "end_ix": 84,
                "relations": []
            },
            "21": {
                "tokens": "vasculature",
                "label": "ANAT-DP",
                "start_ix": 85,
                "end_ix": 85,
                "relations": [
                    [
                        "modify",
                        "20"
                    ]
                ]
            },
            "22": {
                "tokens": "engorged",
                "label": "OBS-DA",
                "start_ix": 88,
                "end_ix": 88,
                "relations": [
                    [
                        "located_at",
                        "20"
                    ]
                ]
            },
            "23": {
                "tokens": "minimal",
                "label": "OBS-DP",
                "start_ix": 92,
                "end_ix": 92,
                "relations": [
                    [
                        "modify",
                        "24"
                    ]
                ]
            },
            "24": {
                "tokens": "atelectasis",
                "label": "OBS-DP",
                "start_ix": 93,
                "end_ix": 93,
                "relations": [
                    [
                        "located_at",
                        "25"
                    ]
                ]
            },
            "25": {
                "tokens": "lung",
                "label": "ANAT-DP",
                "start_ix": 96,
                "end_ix": 96,
                "relations": []
            },
            "26": {
                "tokens": "bases",
                "label": "ANAT-DP",
                "start_ix": 97,
                "end_ix": 97,
                "relations": [
                    [
                        "modify",
                        "25"
                    ]
                ]
            },
            "27": {
                "tokens": "lungs",
                "label": "ANAT-DP",
                "start_ix": 100,
                "end_ix": 100,
                "relations": []
            },
            "28": {
                "tokens": "clear",
                "label": "OBS-DP",
                "start_ix": 102,
                "end_ix": 102,
                "relations": [
                    [
                        "located_at",
                        "27"
                    ]
                ]
            },
            "29": {
                "tokens": "large",
                "label": "OBS-DA",
                "start_ix": 105,
                "end_ix": 105,
                "relations": []
            },
            "30": {
                "tokens": "pleural",
                "label": "ANAT-DP",
                "start_ix": 106,
                "end_ix": 106,
                "relations": []
            },
            "31": {
                "tokens": "effusion",
                "label": "OBS-DA",
                "start_ix": 107,
                "end_ix": 107,
                "relations": [
                    [
                        "located_at",
                        "30"
                    ]
                ]
            },
            "32": {
                "tokens": "pneumothorax",
                "label": "OBS-DA",
                "start_ix": 109,
                "end_ix": 109,
                "relations": []
            },
            "33": {
                "tokens": "extreme",
                "label": "ANAT-DP",
                "start_ix": 114,
                "end_ix": 114,
                "relations": [
                    [
                        "modify",
                        "35"
                    ]
                ]
            },
            "34": {
                "tokens": "right",
                "label": "ANAT-DP",
                "start_ix": 115,
                "end_ix": 115,
                "relations": [
                    [
                        "modify",
                        "35"
                    ]
                ]
            },
            "35": {
                "tokens": "apex",
                "label": "ANAT-DP",
                "start_ix": 116,
                "end_ix": 116,
                "relations": []
            },
            "36": {
                "tokens": "slightly",
                "label": "OBS-DP",
                "start_ix": 118,
                "end_ix": 118,
                "relations": [
                    [
                        "modify",
                        "37"
                    ]
                ]
            },
            "37": {
                "tokens": "obscured",
                "label": "OBS-DP",
                "start_ix": 119,
                "end_ix": 119,
                "relations": [
                    [
                        "located_at",
                        "35"
                    ]
                ]
            },
            "38": {
                "tokens": "patient 's chin",
                "label": "OBS-DP",
                "start_ix": 122,
                "end_ix": 124,
                "relations": [
                    [
                        "modify",
                        "37"
                    ]
                ]
            },
            "39": {
                "tokens": "displaced",
                "label": "OBS-DA",
                "start_ix": 131,
                "end_ix": 131,
                "relations": [
                    [
                        "modify",
                        "40"
                    ]
                ]
            },
            "40": {
                "tokens": "fractures",
                "label": "OBS-DA",
                "start_ix": 132,
                "end_ix": 132,
                "relations": []
            },
            "41": {
                "tokens": "acute",
                "label": "OBS-DA",
                "start_ix": 137,
                "end_ix": 137,
                "relations": [
                    [
                        "modify",
                        "43"
                    ]
                ]
            },
            "42": {
                "tokens": "osseous",
                "label": "ANAT-DP",
                "start_ix": 138,
                "end_ix": 138,
                "relations": []
            },
            "43": {
                "tokens": "abnormality",
                "label": "OBS-DA",
                "start_ix": 139,
                "end_ix": 139,
                "relations": [
                    [
                        "located_at",
                        "42"
                    ]
                ]
            },
            "44": {
                "tokens": "Gaseous",
                "label": "OBS-DP",
                "start_ix": 143,
                "end_ix": 143,
                "relations": [
                    [
                        "modify",
                        "45"
                    ]
                ]
            },
            "45": {
                "tokens": "distension",
                "label": "OBS-DP",
                "start_ix": 144,
                "end_ix": 144,
                "relations": [
                    [
                        "located_at",
                        "46"
                    ]
                ]
            },
            "46": {
                "tokens": "stomach",
                "label": "ANAT-DP",
                "start_ix": 147,
                "end_ix": 147,
                "relations": []
            },
            "47": {
                "tokens": "Standard",
                "label": "OBS-DP",
                "start_ix": 155,
                "end_ix": 155,
                "relations": [
                    [
                        "modify",
                        "48"
                    ]
                ]
            },
            "48": {
                "tokens": "positioning",
                "label": "OBS-DP",
                "start_ix": 156,
                "end_ix": 156,
                "relations": [
                    [
                        "modify",
                        "50"
                    ]
                ]
            },
            "49": {
                "tokens": "endotracheal",
                "label": "OBS-DP",
                "start_ix": 159,
                "end_ix": 159,
                "relations": [
                    [
                        "modify",
                        "50"
                    ]
                ]
            },
            "50": {
                "tokens": "tube",
                "label": "OBS-DP",
                "start_ix": 160,
                "end_ix": 160,
                "relations": []
            },
            "51": {
                "tokens": "Enteric",
                "label": "OBS-DP",
                "start_ix": 164,
                "end_ix": 164,
                "relations": [
                    [
                        "modify",
                        "52"
                    ]
                ]
            },
            "52": {
                "tokens": "tube",
                "label": "OBS-DP",
                "start_ix": 165,
                "end_ix": 165,
                "relations": []
            },
            "53": {
                "tokens": "tip",
                "label": "OBS-DP",
                "start_ix": 166,
                "end_ix": 166,
                "relations": [
                    [
                        "modify",
                        "52"
                    ],
                    [
                        "located_at",
                        "55"
                    ]
                ]
            },
            "54": {
                "tokens": "distal",
                "label": "ANAT-DP",
                "start_ix": 172,
                "end_ix": 172,
                "relations": [
                    [
                        "modify",
                        "55"
                    ]
                ]
            },
            "55": {
                "tokens": "esophagus",
                "label": "ANAT-DP",
                "start_ix": 173,
                "end_ix": 173,
                "relations": []
            },
            "56": {
                "tokens": "advanced by at least 11 cm",
                "label": "OBS-DP",
                "start_ix": 177,
                "end_ix": 182,
                "relations": [
                    [
                        "modify",
                        "53"
                    ]
                ]
            },
            "57": {
                "tokens": "Gaseous",
                "label": "OBS-DP",
                "start_ix": 184,
                "end_ix": 184,
                "relations": [
                    [
                        "modify",
                        "58"
                    ]
                ]
            },
            "58": {
                "tokens": "distention",
                "label": "OBS-DP",
                "start_ix": 185,
                "end_ix": 185,
                "relations": [
                    [
                        "located_at",
                        "59"
                    ]
                ]
            },
            "59": {
                "tokens": "stomach",
                "label": "ANAT-DP",
                "start_ix": 188,
                "end_ix": 188,
                "relations": []
            },
            "60": {
                "tokens": "focal",
                "label": "OBS-DA",
                "start_ix": 196,
                "end_ix": 196,
                "relations": [
                    [
                        "modify",
                        "61"
                    ]
                ]
            },
            "61": {
                "tokens": "consolidation",
                "label": "OBS-DA",
                "start_ix": 197,
                "end_ix": 197,
                "relations": []
            }
        }
    }
  """

prompt = """
Create a JSON file with 5 full-text radiology reports. Take a look at {format} for examples. It should generate in that format.
Each report should include the following fields:
- text
- entities: maps to a dictionary of entities labeled in the report. Each entity has an “entity_id,” which is a unique identifier of the entity in the report. “entity_id” maps to a dictionary with the following keys: 
  1. “tokens” maps to one or more tokens that make up an entity.
  2. “labels” maps to one of the four entities defined by the schema. The labels will consist of two broad entity types: Observation and Anatomy. The Observation entity type includes three uncertainty levels: Definitely Present, Uncertain, and Definitely Absent. Thus, in total, we have four entities, which are labeled as “ANAT-DP,” “OBS-DP,” “OBS-U,” and “OBS-DA.” 
  3. “start_ix” maps to the index of the entity’s first token, using zero-based indexing.
  4. “end_ix” maps to the index of the entity’s last token, using zero-based indexing.
  5. "relations" maps to a list of relations for which the entity is the subject. Each relation is a tuple of ("relation_type", "object_id"). The "relation_type" is one of the three relations defined by the schema. The "object_id" is the id of the other entity in the relation. The three types of relations are :
        - suggestive_of (Observation, Observation) is a relation between two Observation entities indicating that the status of the second Observation is inferred from that of the first Observation.
        - located_at (Observation, Anatomy) is a relation between an Observation entity and an Anatomy entity indicating that the Observation is related to the Anatomy. While located_at often refers to location, it can also be used to describe other relations between an Observation and an Anatomy.
        - modify (Observation, Observation) or (Anatomy, Anatomy) is a relation between two Observation entities or two Anatomy entities indicating that the first entity modifies the scope of, or quantifies the degree of, the second entity. As a result, all Observation modifiers are annotated as Observation entities, and all Anatomy modifiers are annotated as Anatomy entities.

**Additional Instruction**: Avoid labeling insignificant words or common stop words, such as “in,” “the,” “of,” and other similar non-medical terms. Only assign labels to that are medically relevant to anatomy or observations. Do not label anything in the text. Label each word separately.
Ensure the JSON uses numeric keys for each entity (e.g., "1", "2") in `entities` and avoids additional nested structures. Each entity should be an individual dictionary mapped by its numeric key. Only respond with the JSON.
"""

# Run the command using the subprocess module
result = subprocess.run(
    ["ollama", "run", "llama3.1"],  # Replace with correct model name in Ollama
    input=prompt,
    text=True,
    capture_output=True
)

if result.returncode == 0:
    # Attempt to parse only the JSON portion of the response
    try:
        output_text = result.stdout.strip()  # Remove any extraneous whitespace
        start_idx = output_text.find('{')
        end_idx = output_text.rfind('}') + 1
        output_json = json.loads(output_text[start_idx:end_idx])

        # Save to a file
        output_path = "RadAnnotate/data_augmentation/synthetic_radiology_reports.json"
        with open(output_path, "w") as json_file:
            json.dump(output_json, json_file, indent=4)
        
        print(f"Output saved to {output_path}")
    except json.JSONDecodeError:
        print("Failed to decode JSON output. Raw output was:\n", result.stdout)
else:
    print("Error in running the subprocess:", result.stderr)
