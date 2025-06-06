import json
from collections import defaultdict

def extract_categories_from_json(json_data):
    """
    Extract unique categories and l2_categories from JSON data
    and show their relationships
    """
    # If json_data is a string, parse it
    if isinstance(json_data, str):
        data = json.loads(json_data)
    else:
        data = json_data
    
    # Dictionary to store category -> l2_categories mapping
    category_mapping = defaultdict(set)
    
    # Sets to store unique values
    unique_categories = set()
    unique_l2_categories = set()
    
    # Process each item in the JSON array
    for item in data:
        category = item.get('category', '')
        l2_category = item.get('l2_category', '')
        
        if category:
            unique_categories.add(category)
        if l2_category:
            unique_l2_categories.add(l2_category)
        
        # Map category to l2_category
        if category and l2_category:
            category_mapping[category].add(l2_category)
    
    return unique_categories, unique_l2_categories, category_mapping

def print_results(unique_categories, unique_l2_categories, category_mapping):
    """Print the results in a formatted way"""
    print("=" * 60)
    print("UNIQUE CATEGORIES:")
    print("=" * 60)
    for i, category in enumerate(sorted(unique_categories), 1):
        print(f"{i}. {category}")
    
    print("\n" + "=" * 60)
    print("UNIQUE L2_CATEGORIES:")
    print("=" * 60)
    for i, l2_category in enumerate(sorted(unique_l2_categories), 1):
        print(f"{i}. {l2_category}")
    
    print("\n" + "=" * 60)
    print("CATEGORY -> L2_CATEGORY MAPPING:")
    print("=" * 60)
    for category in sorted(category_mapping.keys()):
        l2_cats = sorted(list(category_mapping[category]))
        print(f"\n{category}:")
        for l2_cat in l2_cats:
            print(f"   └── {l2_cat}")

# Example usage:
if __name__ == "__main__":
    # Load from file
    with open('datasets/MME-RealWorld-Lite/json/label.json', 'r') as f:
        json_data = json.load(f)

    categories, l2_categories, mapping = extract_categories_from_json(json_data)
    print_results(categories, l2_categories, mapping)
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("=" * 60)
    print(f"Total unique categories: {len(categories)}")
    print(f"Total unique l2_categories: {len(l2_categories)}")