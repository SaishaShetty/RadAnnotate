import subprocess

def query_ollama(report_text):
    # Define the prompt with a placeholder for the clinical report input
    prompt = f"""
    Your task is to extract anatomy terms exactly as they appear in the report. Do not modify, infer, or shorten terms in any way. Only output terms that appear word-for-word in the text. Go through each individual word.

    [THOUGHT_START]The terms should solely be an anatomy in the body. Give reason for why you labelled them as an anatomy. If your reason does not indicate it is a anatomy part but a condition related to the anatomy, do not label it. You must not label any anatomy condition as anatomy by mistake.[THOUGHT_END]

    Report: {report_text}

    Output:
    [ANATOMY] term1, term2  

  
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
