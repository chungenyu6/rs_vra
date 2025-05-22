import json

# Load generative_query.json
with open("./datasets/AMBER100/query/generative_query.json", "r") as f:
    generative_query = json.load(f)

# Extract the list of IDs from generative_query.json
id_list = {entry["id"]: entry["image"] for entry in generative_query}

# Load annotations.json
with open("./datasets/AMBER100/utils/annotations.json", "r") as f:
    annotations = json.load(f)

# Extract the hallucinated objects based on matching IDs
data = []
for entry in annotations:
    if entry["id"] in id_list:
        data.append({
            "clean_img": id_list[entry["id"]],
            "adv_img": id_list[entry["id"]],
            "target_text": entry["hallu"]
        })

# Print or save the output
print(data)

# Optionally, save to a new JSON file
with open("./datasets/AMBER100/utils/setup_AdvIllusions.json", "w") as f:
    json.dump(data, f, indent=4)