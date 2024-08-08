import os
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOpenAI

# Set your OpenAI API key
os.environ['OPENAI_API_KEY'] = 'sk-proj-NvqIphRNzF9iGBurzBy2T3BlbkFJ1yFpMCXSYc2ubsrt8Xh1'

# Define the prompt template
template = """
You are an expert in medical text annotation. Please annotate the following medical text using the provided labels.

Labels and Definitions:
- CHAN: Mark various types of change.
- CHAN-NC: Indicate a lack of change since the prior study.
- CHAN-CON: Mark changes in the state of the patient's medical conditions.
    - CHAN-CON-AP: Indicate a new adverse medical condition.
    - CHAN-CON-WOR: Indicate worsening of a condition.
    - CHAN-CON-IMP: Indicate improvement in a condition.
    - CHAN-CON-RES: Indicate resolution of a condition.
- CHAN-DEV: Mark changes related to medical devices.
    - CHAN-DEV-AP: Indicate a new medical device or tool.
    - CHAN-DEV-PLACE: Indicate a change in device placement.
    - CHAN-DEV-DISA: Indicate a removed medical device.
- ANAT: Mark anatomical body parts.
    - ANAT-DP: Mark all mentions of body parts or anatomical locations.
- OBS: Mark observations from radiology images.
    - OBS-DP: Indicate observations recorded with high certainty.
    - OBS-U: Indicate uncertain observations.
    - OBS-DA: Indicate observations which could be excluded.

Examples:
- Text: "Moderately severe bibasilar atelectasis persists."
  Annotation: CHAN-NC

- Text: "There is also a new left basilar opacity blunting the lateral costophrenic angle."
  Annotation: CHAN-CON-AP

- Text: "Mild to moderate diffuse pulmonary edema is slightly worse."
  Annotation: CHAN-CON-WOR

- Text: "Compared to the most recent study, there is improvement in the mild pulmonary edema and decrease in the small left pleural effusion."
  Annotation: CHAN-CON-IMP

- Text: "Indistinct superior segment left lower lobe opacities have resolved."
  Annotation: CHAN-CON-RES

- Text: "The patient has received the new feeding tube."
  Annotation: CHAN-DEV-AP

- Text: "Left pleural drain has been advanced to the left apex."
  Annotation: CHAN-DEV-PLACE

- Text: "In the interval, the patient has been extubated."
  Annotation: CHAN-DEV-DISA

- Text: "The left lung is essentially clear."
  Annotation: ANAT-DP

- Text: "There is moderate cardiomegaly."
  Annotation: OBS-DP

- Text: "Infection cannot be excluded."
  Annotation: OBS-U

- Text: "There is no pneumothorax."
  Annotation: OBS-DA

Text to Annotate:
"{text}"
"""

# Initialize the OpenAI LLM
llm = ChatOpenAI(model='gpt-4')

prompt = PromptTemplate(template=template, input_variables=['text'])


chain = LLMChain(llm=llm, prompt=prompt)

def annotate_text_with_gpt4(text):
    
    annotation_result = chain.run(text=text)
    return annotation_result

# Example text report
text_report = """
FINAL REPORT EXAMINATION : CHEST ( PORTABLE AP ) INDICATION : History : ___ F with ett placement TECHNIQUE : Upright AP view of the chest COMPARISON : None . Patient is currently listed as EU critical . FINDINGS : Endotracheal tube tip terminates approximately a 4.6 cm from the carina . Enteric tube tip terminates within the distal esophagus and should be advanced by at least 11 cm . Heart size is normal . The mediastinal and hilar contours are grossly unremarkable . Pulmonary vasculature is not engorged . Apart from minimal atelectasis in the lung bases , the lungs appear clear . No large pleural effusion or pneumothorax is present though the extreme right apex is slightly obscured by the patient 's chin projecting over this area . No displaced fractures are present . No acute osseous abnormality is seen . Gaseous distension of the stomach is noted . IMPRESSION : 1 . Standard positioning of the endotracheal tube . 2 . Enteric tube tip is suboptimally located within the distal esophagus and should be advanced by at least 11 cm . Gaseous distention of the stomach is also noted . 3 . No focal consolidation .
"""

# Get annotations
annotations = annotate_text_with_gpt4(text_report)

# Print annotations
print("Annotated Text:")
print(annotations)
