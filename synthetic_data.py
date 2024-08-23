import openai
import json
import random
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableSequence

# Define your API key (uncomment and use your own API key)
# openai.api_key = 'your-api-key-here'

# List of five labels
label_list = [
    "ANAT-DP",
    "OBS-DP",
    "OBS-DA",
    "ANAT-DA",
    "OBS-DD"
]

# Define the prompt template
template = """
You are a radiologist tasked with generating synthetic clinical radiology chest reports.
The reports should contain realistic clinical findings and impressions that mention some of the labels from the provided list, but the labels themselves should not appear in the report.

The labels to include are: {labels}.

The report should describe findings associated with the labels. Do not include the labels in the report text.

After generating the report, create a separate entity list in this format:
"label: word"
where each label is mapped to the word or phrase in the report that corresponds to that label.
"""

# Create the prompt template
prompt = PromptTemplate(input_variables=["labels"], template=template)

# Initialize the LLM
llm = ChatOpenAI(model='gpt-4')

# Create the chain with the prompt and model
chain = RunnableSequence(prompt | llm)

# Generate and store 5 reports
reports = []
for _ in range(5):
    # Randomly select 2-3 labels for each report
    selected_labels = random.sample(label_list, random.randint(2, 3))
    labels_str = ', '.join(selected_labels)
    
    # Generate the report
    inputs = {"labels": labels_str}
    generated_report = chain.invoke(inputs)
    
    # Extract the text from the AIMessage object
    report_text = generated_report.content.strip()

    # Initialize the entity dictionary
    entities = {}

    # Split the output into the report and the entity list
    if "Entity list:" in report_text:
        report, entity_list = report_text.split("\nEntity list:\n", 1)
        # Process the entity list into a dictionary
        for line in entity_list.strip().split("\n"):
            if ": " in line:
                label, word = line.split(": ", 1)
                entities[label.strip()] = word.strip()
    else:
        report = report_text

    # Append the report and entities to the list
    reports.append({
        "report": report.strip(),
        "entities": entities,
        "included_labels": selected_labels
    })

# Store the reports in a JSON file
with open("synthetic_clinical_reports.json", "w") as f:
    json.dump(reports, f, indent=4)

print("Reports generated and saved to synthetic_clinical_reports.json")
