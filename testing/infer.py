import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

base_model = "/home/spshetty/RadAnnotate/Meta-Llama-3-8B"
new_model = "/home/spshetty/RadAnnotate/finetuned_models/instr/epoch_10/checkpoint-1060"

# QLoRA config for loading in 4-bit quantization (optional)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# quantization config (if necessary)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)


tokenizer = AutoTokenizer.from_pretrained(base_model)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if len(tokenizer) != model.get_input_embeddings().num_embeddings:
    model.resize_token_embeddings(len(tokenizer))

example = """{{
"Report (Input)": "FINAL REPORT INDICATION : ___ year old man with 2 week h / o cough and malaise . Slow to improve after treatment with antibiotics / / r / o infiltrate TECHNIQUE : Chest PA and lateral COMPARISON : Chest radiograph ___ FINDINGS : The lung volumes are normal . Mild cardiomegaly which is stable . Normal hilar and mediastinal structures . No pneumonia , no pulmonary edema . No pleural effusions . Status post CABG with aligned median sternotomy wires and normal location of surgical clips . Status post right lung surgery with surgical material seen . IMPRESSION : Mild cardiomegaly . No evidence of pneumonia .",
"Annotated Report (Output)" : {
        "1": {
          "tokens": "lung",
          "label": "ANAT-DP",
          "start_ix": 45,
          "end_ix": 45,
          "relations": []
        },
        "2": {
          "tokens": "volumes",
          "label": "ANAT-DP",
          "start_ix": 46,
          "end_ix": 46,
          "relations": [
            [
              "modify",
              "1"
            ]
          ]
        },
        "3": {
          "tokens": "normal",
          "label": "OBS-DP",
          "start_ix": 48,
          "end_ix": 48,
          "relations": [
            [
              "located_at",
              "1"
            ]
          ]
        },
        "4": {
          "tokens": "Mild",
          "label": "OBS-DP",
          "start_ix": 50,
          "end_ix": 50,
          "relations": [
            [
              "modify",
              "5"
            ]
          ]
        },
        "5": {
          "tokens": "cardiomegaly",
          "label": "OBS-DP",
          "start_ix": 51,
          "end_ix": 51,
          "relations": []
        },
        "6": {
          "tokens": "stable",
          "label": "OBS-DP",
          "start_ix": 54,
          "end_ix": 54,
          "relations": [
            [
              "modify",
              "5"
            ]
          ]
        },
        "7": {
          "tokens": "Normal",
          "label": "OBS-DP",
          "start_ix": 56,
          "end_ix": 56,
          "relations": [
            [
              "located_at",
              "8"
            ],
            [
              "located_at",
              "9"
            ]
          ]
        },
        "8": {
          "tokens": "hilar",
          "label": "ANAT-DP",
          "start_ix": 57,
          "end_ix": 57,
          "relations": []
        },
        "9": {
          "tokens": "mediastinal",
          "label": "ANAT-DP",
          "start_ix": 59,
          "end_ix": 59,
          "relations": []
        },
        "10": {
          "tokens": "structures",
          "label": "ANAT-DP",
          "start_ix": 60,
          "end_ix": 60,
          "relations": [
            [
              "modify",
              "8"
            ],
            [
              "modify",
              "9"
            ]
          ]
        },
        "11": {
          "tokens": "pneumonia",
          "label": "OBS-DA",
          "start_ix": 63,
          "end_ix": 63,
          "relations": []
        },
        "12": {
          "tokens": "pulmonary",
          "label": "ANAT-DP",
          "start_ix": 66,
          "end_ix": 66,
          "relations": []
        },
        "13": {
          "tokens": "edema",
          "label": "OBS-DA",
          "start_ix": 67,
          "end_ix": 67,
          "relations": [
            [
              "located_at",
              "12"
            ]
          ]
        },
        "14": {
          "tokens": "pleural",
          "label": "ANAT-DP",
          "start_ix": 70,
          "end_ix": 70,
          "relations": []
        },
        "15": {
          "tokens": "effusions",
          "label": "OBS-DA",
          "start_ix": 71,
          "end_ix": 71,
          "relations": [
            [
              "located_at",
              "14"
            ]
          ]
        },
        "16": {
          "tokens": "post",
          "label": "OBS-DP",
          "start_ix": 74,
          "end_ix": 74,
          "relations": [
            [
              "modify",
              "17"
            ]
          ]
        },
        "17": {
          "tokens": "CABG",
          "label": "OBS-DP",
          "start_ix": 75,
          "end_ix": 75,
          "relations": []
        },
        "18": {
          "tokens": "aligned",
          "label": "OBS-DP",
          "start_ix": 77,
          "end_ix": 77,
          "relations": [
            [
              "modify",
              "21"
            ]
          ]
        },
        "19": {
          "tokens": "median",
          "label": "OBS-DP",
          "start_ix": 78,
          "end_ix": 78,
          "relations": [
            [
              "modify",
              "21"
            ]
          ]
        },
        "20": {
          "tokens": "sternotomy",
          "label": "OBS-DP",
          "start_ix": 79,
          "end_ix": 79,
          "relations": [
            [
              "modify",
              "21"
            ]
          ]
        },
        "21": {
          "tokens": "wires",
          "label": "OBS-DP",
          "start_ix": 80,
          "end_ix": 80,
          "relations": []
        },
        "22": {
          "tokens": "normal",
          "label": "OBS-DP",
          "start_ix": 82,
          "end_ix": 82,
          "relations": [
            [
              "modify",
              "25"
            ]
          ]
        },
        "23": {
          "tokens": "location",
          "label": "OBS-DP",
          "start_ix": 83,
          "end_ix": 83,
          "relations": [
            [
              "modify",
              "22"
            ]
          ]
        },
        "24": {
          "tokens": "surgical",
          "label": "OBS-DP",
          "start_ix": 85,
          "end_ix": 85,
          "relations": [
            [
              "modify",
              "25"
            ]
          ]
        },
        "25": {
          "tokens": "clips",
          "label": "OBS-DP",
          "start_ix": 86,
          "end_ix": 86,
          "relations": []
        },
        "26": {
          "tokens": "post",
          "label": "OBS-DP",
          "start_ix": 89,
          "end_ix": 89,
          "relations": [
            [
              "modify",
              "29"
            ]
          ]
        },
        "27": {
          "tokens": "right",
          "label": "ANAT-DP",
          "start_ix": 90,
          "end_ix": 90,
          "relations": [
            [
              "modify",
              "28"
            ]
          ]
        },
        "28": {
          "tokens": "lung",
          "label": "ANAT-DP",
          "start_ix": 91,
          "end_ix": 91,
          "relations": []
        },
        "29": {
          "tokens": "surgery",
          "label": "OBS-DP",
          "start_ix": 92,
          "end_ix": 92,
          "relations": [
            [
              "located_at",
              "28"
            ]
          ]
        },
        "30": {
          "tokens": "surgical",
          "label": "OBS-DP",
          "start_ix": 94,
          "end_ix": 94,
          "relations": [
            [
              "modify",
              "31"
            ]
          ]
        },
        "31": {
          "tokens": "material",
          "label": "OBS-DP",
          "start_ix": 95,
          "end_ix": 95,
          "relations": []
        },
        "32": {
          "tokens": "Mild",
          "label": "OBS-DP",
          "start_ix": 100,
          "end_ix": 100,
          "relations": [
            [
              "modify",
              "33"
            ]
          ]
        },
        "33": {
          "tokens": "cardiomegaly",
          "label": "OBS-DP",
          "start_ix": 101,
          "end_ix": 101,
          "relations": []
        },
        "34": {
          "tokens": "pneumonia",
          "label": "OBS-DA",
          "start_ix": 106,
          "end_ix": 106,
          "relations": []
        }
      },
    {
    "Report (Input)": "FINAL REPORT INDICATION : ___ year old woman with severe COPD , s / p PEA arrest here with respiratory failure and complete heart block , now s / p pacemaker placement / / eval for pacemaker placement , interval pulmonary change COMPARISON : The comparison is made with prior studies including ___ . IMPRESSION : The endotracheal tube tip is 6 cm above the carina . Nasogastric tube tip is beyond the GE junction and off the edge of the film . A left central line is present in the tip is in the mid SVC . A pacemaker is noted on the right in the lead projects over the right ventricle . There is probable scarring in both lung apices . There are no new areas of consolidation . There is upper zone redistribution and cardiomegaly suggesting pulmonary venous hypertension . There is no pneumothorax .",
    "Annotated Report (Output)" : {
        "1": {
          "tokens": "endotracheal",
          "label": "OBS-DP",
          "start_ix": 57,
          "end_ix": 57,
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
          "start_ix": 58,
          "end_ix": 58,
          "relations": [
            [
              "located_at",
              "5"
            ]
          ]
        },
        "3": {
          "tokens": "tip",
          "label": "OBS-DP",
          "start_ix": 59,
          "end_ix": 59,
          "relations": [
            [
              "modify",
              "2"
            ]
          ]
        },
        "4": {
          "tokens": "6 cm above",
          "label": "OBS-DP",
          "start_ix": 61,
          "end_ix": 63,
          "relations": [
            [
              "located_at",
              "5"
            ]
          ]
        },
        "5": {
          "tokens": "carina",
          "label": "ANAT-DP",
          "start_ix": 65,
          "end_ix": 65,
          "relations": []
        },
        "6": {
          "tokens": "Nasogastric",
          "label": "OBS-DP",
          "start_ix": 67,
          "end_ix": 67,
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
          "start_ix": 68,
          "end_ix": 68,
          "relations": []
        },
        "8": {
          "tokens": "tip",
          "label": "OBS-DP",
          "start_ix": 69,
          "end_ix": 69,
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
          "tokens": "beyond",
          "label": "ANAT-DP",
          "start_ix": 71,
          "end_ix": 71,
          "relations": [
            [
              "modify",
              "10"
            ]
          ]
        },
        "10": {
          "tokens": "GE junction",
          "label": "ANAT-DP",
          "start_ix": 73,
          "end_ix": 74,
          "relations": []
        },
        "11": {
          "tokens": "left",
          "label": "OBS-DP",
          "start_ix": 84,
          "end_ix": 84,
          "relations": [
            [
              "modify",
              "13"
            ]
          ]
        },
        "12": {
          "tokens": "central",
          "label": "OBS-DP",
          "start_ix": 85,
          "end_ix": 85,
          "relations": [
            [
              "modify",
              "13"
            ]
          ]
        },
        "13": {
          "tokens": "line",
          "label": "OBS-DP",
          "start_ix": 86,
          "end_ix": 86,
          "relations": []
        },
        "14": {
          "tokens": "tip",
          "label": "OBS-DP",
          "start_ix": 91,
          "end_ix": 91,
          "relations": [
            [
              "modify",
              "13"
            ],
            [
              "located_at",
              "16"
            ]
          ]
        },
        "15": {
          "tokens": "mid",
          "label": "ANAT-DP",
          "start_ix": 95,
          "end_ix": 95,
          "relations": [
            [
              "modify",
              "16"
            ]
          ]
        },
        "16": {
          "tokens": "SVC",
          "label": "ANAT-DP",
          "start_ix": 96,
          "end_ix": 96,
          "relations": []
        },
        "17": {
          "tokens": "pacemaker",
          "label": "OBS-DP",
          "start_ix": 99,
          "end_ix": 99,
          "relations": []
        },
        "18": {
          "tokens": "right",
          "label": "OBS-DP",
          "start_ix": 104,
          "end_ix": 104,
          "relations": [
            [
              "modify",
              "17"
            ]
          ]
        },
        "19": {
          "tokens": "lead",
          "label": "OBS-DP",
          "start_ix": 107,
          "end_ix": 107,
          "relations": [
            [
              "modify",
              "17"
            ],
            [
              "located_at",
              "22"
            ]
          ]
        },
        "20": {
          "tokens": "over",
          "label": "ANAT-DP",
          "start_ix": 109,
          "end_ix": 109,
          "relations": [
            [
              "modify",
              "22"
            ]
          ]
        },
        "21": {
          "tokens": "right",
          "label": "ANAT-DP",
          "start_ix": 111,
          "end_ix": 111,
          "relations": [
            [
              "modify",
              "22"
            ]
          ]
        },
        "22": {
          "tokens": "ventricle",
          "label": "ANAT-DP",
          "start_ix": 112,
          "end_ix": 112,
          "relations": []
        },
        "23": {
          "tokens": "scarring",
          "label": "OBS-U",
          "start_ix": 117,
          "end_ix": 117,
          "relations": [
            [
              "located_at",
              "25"
            ]
          ]
        },
        "24": {
          "tokens": "both",
          "label": "ANAT-DP",
          "start_ix": 119,
          "end_ix": 119,
          "relations": [
            [
              "modify",
              "26"
            ]
          ]
        },
        "25": {
          "tokens": "lung",
          "label": "ANAT-DP",
          "start_ix": 120,
          "end_ix": 120,
          "relations": []
        },
        "26": {
          "tokens": "apices",
          "label": "ANAT-DP",
          "start_ix": 121,
          "end_ix": 121,
          "relations": [
            [
              "modify",
              "25"
            ]
          ]
        },
        "27": {
          "tokens": "new",
          "label": "OBS-DA",
          "start_ix": 126,
          "end_ix": 126,
          "relations": [
            [
              "modify",
              "28"
            ]
          ]
        },
        "28": {
          "tokens": "areas",
          "label": "OBS-DA",
          "start_ix": 127,
          "end_ix": 127,
          "relations": [
            [
              "modify",
              "29"
            ]
          ]
        },
        "29": {
          "tokens": "consolidation",
          "label": "OBS-DA",
          "start_ix": 129,
          "end_ix": 129,
          "relations": []
        },
        "30": {
          "tokens": "upper",
          "label": "ANAT-DP",
          "start_ix": 133,
          "end_ix": 133,
          "relations": [
            [
              "modify",
              "31"
            ]
          ]
        },
        "31": {
          "tokens": "zone",
          "label": "ANAT-DP",
          "start_ix": 134,
          "end_ix": 134,
          "relations": []
        },
        "32": {
          "tokens": "redistribution",
          "label": "OBS-DP",
          "start_ix": 135,
          "end_ix": 135,
          "relations": [
            [
              "located_at",
              "31"
            ],
            [
              "suggestive_of",
              "36"
            ]
          ]
        },
        "33": {
          "tokens": "cardiomegaly",
          "label": "OBS-DP",
          "start_ix": 137,
          "end_ix": 137,
          "relations": [
            [
              "suggestive_of",
              "36"
            ]
          ]
        },
        "34": {
          "tokens": "pulmonary",
          "label": "ANAT-DP",
          "start_ix": 139,
          "end_ix": 139,
          "relations": [
            [
              "modify",
              "35"
            ]
          ]
        },
        "35": {
          "tokens": "venous",
          "label": "ANAT-DP",
          "start_ix": 140,
          "end_ix": 140,
          "relations": []
        },
        "36": {
          "tokens": "hypertension",
          "label": "OBS-DP",
          "start_ix": 141,
          "end_ix": 141,
          "relations": [
            [
              "located_at",
              "35"
            ]
          ]
        },
        "37": {
          "tokens": "pneumothorax",
          "label": "OBS-DA",
          "start_ix": 146,
          "end_ix": 146,
          "relations": []
        }
      }
    }
"""
def format_chat_prompt(report):
    return f"""
    Instruction: You are given a clinical radiology report. 
    Follow these steps to annotate the report:

    1. Identify labels: Our schema defines two entity types: Observation and Anatomy. Observations have two uncertainty levels: Definitely Present and Definitely Absent. 
       The entities are: 
       - ANAT-DA (Anatomy - Definitely Absent)
       - ANAT-DP (Anatomy - Definitely Present)
       - OBS-DA (Observation - Definitely Absent)
       - OBS-DP (Observation - Definitely Present)

    2. Label terms relevant to the schema. Avoid labeling terms like "and, or, the."

    3. Identify relationships between entities:
       - **suggestive_of**: One observation suggests another.
       - **located_at**: Links an observation to an anatomy.
       - **modify**: One entity modifies the other.
       Use the sentence context to identify these relationships.

    4. In the Annotated report which is the output make sure to mention the following lables:
      - “tokens” maps to one or more tokens that make up an entity.
      - “labels” maps to one of the four entities defined by the schema.
      - “start_ix” maps to the index of the entity’s first token, using zero-based indexing.
      - “end_ix” maps to the index of the entity’s last token, using zero-based indexing.
      - “relations” maps to a list of relations for which the entity is the subject. Each relation is a tuple of (“relation_type”, “object_id”). The “relation_type” is one of the three relations defined by the schema. The “object_id” is the id of the other entity in the relation.
    
    Example of an annotated report:
    {example}

    Now, annotate the following report:

    Report (Input): {report}
    Annotated Report (Output): """


# Example report input
reports = ["The lungs are clear . Cardiomediastinal and hilar contours are normal . There are no pleural effusions or pneumothorax ."]

#In comparison with study of ___ , the endotracheal tube and Swan - Ganz catheter have been removed . The left chest tube remains in place and there is no evidence of pneumothorax . Mild atelectatic changes are seen at the left base .
prompts = [format_chat_prompt(report) for report in reports]


pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)

#outputs = [pipe(prompt, max_new_tokens=120, do_sample=False, top_k=50, top_p=0.95)[0]["generated_text"] for prompt in prompts]

outputs = [pipe(prompt, max_new_tokens=2048, do_sample=False, temperature=0.0, top_k=50, top_p=0.95)[0]["generated_text"] for prompt in prompts]

for i, generated_text in enumerate(outputs):
    print(f"Original Report {i+1}: {reports[i]}")
    print(f"Annotated Report {i+1}: {generated_text}")
    print("-" * 80)
