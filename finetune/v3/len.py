import json

def calculate_length_of_json(input_file):
    """
    Calculate and print the number of dictionary records in the JSON list.

    Args:
        input_file (str): Path to input JSON file.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)

    num_records = len(data)
    print(f"✅ Total number of records: {num_records}")

    return num_records


# ---- Set your JSON file path ---- #
input_json_file = '/home/spshetty/RadAnnotate/finetune/v3/train_set_v3.json'

# ---- Run the length calculation ---- #
calculate_length_of_json(input_json_file)
