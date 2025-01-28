import json

def reorder_keys_in_json(input_file, output_file):
    with open(input_file, "r") as file:
        data = json.load(file)

    reordered_data = []

    for obj in data:
        if isinstance(obj, dict):  # Process dictionaries only
            reordered_obj = {
                "Report": obj.get("Report", ""),  # Place "Report" first
                "Annotated Report": obj.get("Annotations", {})  # Place "Annotated Report" second
            }
            # Add other keys in their original order
            for key, value in obj.items():
                if key not in reordered_obj:
                    reordered_obj[key] = value
            reordered_data.append(reordered_obj)
        else:
            reordered_data.append(obj)  # Append non-dict items as-is

    with open(output_file, "w") as file:
        json.dump(reordered_data, file, indent=4)


input_file = "corrected_full_relation_testing.json"  # Input JSON file path
output_file = "check1.0.json"
reorder_keys_in_json(input_file, output_file)