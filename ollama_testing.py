import subprocess

def query_ollama(report_text):
    # Define the prompt with a placeholder for the clinical report input
    prompt = f"""
    You are an AI assistant designed to provide detailed, step-by-step responses for clinical report annotation. Analyze the following clinical report and identify relevant labels and relationships.

    **Clinical Report Input:**
    "{report_text}"

    <thinking>
    1. Analyze the clinical report and identify the labels. The labels can be one of the two entities : Observation and Anatomy. These labels can have two uncertainty levels : Definitely Present and Definitely Absent.
        ANAT-DA (Anatomy - Definitely Absent)
        ANAT-DP (Anatomy - Definitely Present)
        OBS-DA (Observation - Definitely Absent)
        OBS-DP (Observation - Definitely Present)

    2. Briefly analyze the report and label terms only relevant to the labels. Avoid labeling terms like “and,or,the”. 

    3. Identify relationship between entities:
        - **suggestive_of**: One observation suggests another.
        - **located_at**: Links an observation to an anatomy
        - **modify**: One entity modifies the other. 
        Use the sentence context to identify the relationship between the entities.

    4. In the annotated report which is the output make sure to mention the following labels:
        - “tokens” maps to one or more tokens that make up an entity.
        - “labels” maps to one of the four entities defined by the schema.
        - “start_ix” maps to the index of the entity’s first token, using zero-based indexing.
        - “end_ix” maps to the index of the entity’s last token, using zero-based indexing.
        - “relations” maps to a list of relations for which the entity is the subject. Each relation is a tuple of (“relation_type”, “object_id”). The “relation_type” is one of the three relations defined by the schema. The “object_id” is the id of the other entity in the relation.

    5. Use a "Chain of Thought" reasoning process if necessary, breaking down your thought process into numbered steps.

    <reflection>
        1. Review each decision for correct labels and relationships.
        2. Adjust any errors or oversights before finalizing the response.
    </reflection>
    </thinking>
    <output>
    {{example}}
    </output>
    """

    
    result = subprocess.run(
    ["ollama", "run", "llama3.1"],
    input=prompt,
    text=True,
    capture_output=True)


    # Print the output
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("Error:", result.stderr)
   
# Example usage with a sample clinical report
report_text = "The lungs are clear . Cardiomediastinal and hilar contours are normal . There are no pleural effusions or pneumothorax ."
query_ollama(report_text)
