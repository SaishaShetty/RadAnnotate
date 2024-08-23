from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

report = "The lungs are clear. Cardiomediastinal and hilar contours are normal. There are no pleural effusions or pneumothorax."

template = """

Task: You are given a clinical radiology report. 

Our schema defines two broad entity types: Observation and Anatomy. The Observation entity type includes two uncertainty levels: Definitely Present and Definitely Absent. Thus, in total, we have four entities, which are labeled as “ANAT-DP”, “OBS-DP”, “OBS-U”, and “OBS-DA”. Our schema defines three relations between entities, which are labeled as “suggestive_of”, “located_at”, and “modify”.
Annotate the data using the following four entities .
Output Format::
1. ANAT-DA: Anatomy - Definitely Absent  #Abscence of an anatomy part
2. ANAT-DP: Anatomy - Definitely Present  #Presence of an anatomy part
3. OBS-DA: Observation - Definitely Absent  #Abscence of an observation
4. OBS-DP: Observation - Definitely Present  #Presence of an observation

Entity Relations (Schema):

1. suggestive_of (Observation, Observation): Indicates that the status of the second Observation is inferred from the first Observation.  
2. located_at (Observation, Anatomy): Links an Observation entity to an Anatomy entity, indicating their relation. While it often refers to location, it can describe other connections as well.  
3. modify (Observation, Observation) / (Anatomy, Anatomy): Indicates that the first entity modifies or quantifies the degree of the second entity. Modifiers of Observations are annotated as Observation entities, and modifiers of Anatomy as Anatomy entities.

Instructions:  
Annotate the report with the labels and specify any relationships among them using the above schema. Ensure that each label corresponds to individual word tokens from the report. Dont label the whole sentence. It should be word tokens.
Label each word seperately be it observation or anatomy and then mention the relation with its label id. 

Label ID: <ID>
Labels:
{{
    "token" : "<word that is labelled>"
    "label": "<label type>"
}}

Relations:
{{
    (if any)
    "relation": "<label ID in relation>"
}}

Clinical Report:  
{report}

"""
prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3.1")

chain = prompt | model
response = chain.invoke({"report": report})

print(response)