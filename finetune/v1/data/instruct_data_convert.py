import json

# Function to convert input JSON to the desired format
def convert_json_format(input_json):
    converted_data = []

    if isinstance(input_json, list):  # If input is a list of dictionaries
        for details in input_json:
            if isinstance(details, dict):  # Ensure details is a dictionary
                instruction = (
                    "Annotate the given clinical radiology report by identifying relevant "
                    "entities (ANAT-DP, ANAT-DA, OBS-DP, OBS-DA) and their relations "
                    "(suggestive_of, located_at, modify). Output the annotations in JSON format."
                )
                text = details.get("Report", "").strip()  # Extract report text
                entities = details.get("Annotated Report", {})  # Extract annotations

                # Create structured output
                output_entry = {
                    "instruction": instruction,
                    "input": text,
                    "output": entities
                }

                converted_data.append(output_entry)
            else:
                print(f"Skipping an entry due to unexpected format: {details}")
    
    elif isinstance(input_json, dict):  # Handle dictionary input (old format)
        for filename, details in input_json.items():
            if isinstance(details, dict):
                instruction = (
                    "Annotate the given clinical radiology report by identifying relevant "
                    "entities (ANAT-DP, ANAT-DA, OBS-DP, OBS-DA) and their relations "
                    "(suggestive_of, located_at, modify). Output the annotations in JSON format."
                )
                text = details.get("text", "").strip()
                entities = details.get("entities", {})

                output_entry = {
                    "instruction": instruction,
                    "input": text,
                    "output": entities
                }

                converted_data.append(output_entry)
            else:
                print(f"Skipping {filename}: Unexpected format for details.")

    else:
        print("Error: Input JSON is neither a dictionary nor a list of dictionaries.")

    return converted_data

# Function to read, convert, and save the JSON file
def process_json_file(input_file, output_file):
    try:
        # Read input JSON file
        with open(input_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        # Convert the JSON data
        converted_data = convert_json_format(input_data)

        # Write the converted data to the output file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(converted_data, f, indent=4)

        print(f"Successfully processed and saved to {output_file}")

    except json.JSONDecodeError:
        print(f"Error: The file {input_file} contains invalid JSON.")
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# Usage
input_file = "/home/spshetty/RadAnnotate/finetune/v1/data/data_combined.json"  # Replace with actual input file path
output_file = "finetune_instr_data.json"  # Replace with desired output file path

# Process the file
process_json_file(input_file, output_file)
