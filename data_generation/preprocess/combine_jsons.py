import json

def combine_json_files(file1, file2, output_file):
    # Load the first JSON file
    with open(file1, 'r') as f1:
        data1 = json.load(f1)

    # Load the second JSON file
    with open(file2, 'r') as f2:
        data2 = json.load(f2)

    # Combine the two lists
    combined_data = data1 + data2

    # Write the combined data to a new file
    with open(output_file, 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)

# Example usage
file1 = "finetune_train.json"  # Path to the first JSON file
file2 = "/home/spshetty/RadAnnotate/data_generation/mixtral/outputs/synthetic_data.json"  # Path to the second JSON file
output_file = "finetune_data_combined.json"  # Output file path

combine_json_files(file1, file2, output_file)
print("Files combined successfully!")
