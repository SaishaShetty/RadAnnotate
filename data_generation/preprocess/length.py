import json

# Load JSON from a file
with open("/home/spshetty/RadAnnotate/data_generation/mixtral/outputs/try.json", "r") as file:
    data = json.load(file)  # This will load the JSON as a Python list if it's in the format [{},{}]

# Check the number of dictionaries in the list
num_dicts = len(data)
print(f"Number of dictionaries: {num_dicts}")
