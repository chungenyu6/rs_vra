import json
import random
import os
import shutil

# [ ] specify input and output file paths
N = 100
input_file = "/home/vol-llm/datasets/AMBER/data/query/query_generative.json"
output_file = "./query/generative_query.json"
image_source_folder = "/home/vol-llm/datasets/AMBER/data/image"
image_destination_folder = "image"

# Load the JSON data
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Ensure N does not exceed the number of available samples
N = min(N, len(data))

# Sample N entries based on the "id" field
sampled_data = random.sample(data, N)

# Save the sampled data in the same format
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, indent=4, ensure_ascii=False)

# Ensure destination folder exists
os.makedirs(image_destination_folder, exist_ok=True)

# Copy the sampled images to the new folder
for item in sampled_data:
    image_name = item["image"]
    source_path = os.path.join(image_source_folder, image_name)
    destination_path = os.path.join(image_destination_folder, image_name)
    
    if os.path.exists(source_path):
        shutil.copy(source_path, destination_path)
    else:
        print(f"Warning: {source_path} not found!")

print(f"Sampled {N} entries saved to {output_file} and images copied to {image_destination_folder}")