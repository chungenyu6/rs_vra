import json
from collections import defaultdict

def extract_l2_category_indices(json_data):
    """
    Extract indices for each l2_category from JSON data
    Returns a dictionary mapping l2_category -> list of indices
    """
    # If json_data is a string, parse it
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Dictionary to store l2_category -> indices mapping
    l2_category_indices = defaultdict(list)
    
    # Process each item in the JSON array
    for item in data:
        index = item.get('index')
        l2_category = item.get('l2_category', '')
        
        # Add index to the corresponding l2_category
        if l2_category and index is not None:
            l2_category_indices[l2_category].append(index)
    
    # Sort indices for each l2_category
    for l2_cat in l2_category_indices:
        l2_category_indices[l2_cat].sort()
    
    return dict(l2_category_indices)

def print_l2_category_indices(l2_category_indices):
    """Print l2_category indices in the requested format"""
    print("L2_CATEGORY INDICES:")
    print("=" * 50)
    
    # Sort l2_categories alphabetically for consistent output
    for l2_category in sorted(l2_category_indices.keys()):
        indices = l2_category_indices[l2_category]
        indices_str = ', '.join(map(str, indices))
        print(f"{l2_category} = [{indices_str}]")
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("=" * 50)
    for l2_category in sorted(l2_category_indices.keys()):
        count = len(l2_category_indices[l2_category])
        print(f"{l2_category}: {count} items")

def save_l2_category_indices_to_json(l2_category_indices, output_file='l2_category_indices.json'):
    """Save l2_category indices to JSON file in the requested format
    
    Args:
        l2_category_indices: Dictionary of l2_category -> indices
        output_file: Output JSON file name
        compact: If True, use compact formatting (arrays on single lines)
    """
    # Convert to the requested format: list of dictionaries
    output_data = []
    
    # Sort l2_categories alphabetically for consistent output
    for l2_category in sorted(l2_category_indices.keys()):
        indices = l2_category_indices[l2_category]
        output_data.append({l2_category: indices})
    
    # Save to JSON file with different formatting options
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, separators=(',', ': '))
    
    print(f"\nL2 category indices saved to: {output_file}")
    return output_file

if __name__ == "__main__":
    
    # Load from file and save indices to JSON
    label_file = "datasets/MME-RealWorld-Lite/json/label.json"
    with open(label_file, 'r') as f:
        json_data = json.load(f)

    l2_indices = extract_l2_category_indices(json_data)
    print_l2_category_indices(l2_indices)
    
    # Save to JSON file
    output_file = "datasets/MME-RealWorld-Lite/l2_category_indices.json"
    save_l2_category_indices_to_json(l2_indices, output_file)

