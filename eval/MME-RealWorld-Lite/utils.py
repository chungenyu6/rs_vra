import json


def load_l2_category_indices_from_json(json_file):
    """Load l2_category indices from JSON file and convert back to dictionary"""
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Convert back to dictionary format for easy access
    result = {}
    for item in data:
        result.update(item)
    
    return result

def get_indices(l2_category, sample):

    sample = int(sample)

    # Load all the l2_category indices
    json_file = "datasets/MME-RealWorld-Lite/l2_category_indices.json"
    all_indices = load_l2_category_indices_from_json(json_file)

    # Get the indices for the requested l2_category
    l2_category_indices = all_indices.get(l2_category, [])

    return l2_category_indices[:sample]

def assemble_question(question, options):

    msg = """\
    Question: {question}
    The choices are listed below: 
    {choice_a}
    {choice_b}
    {choice_c}
    {choice_d}
    {choice_e}
    Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. 
    The best answer is:
    """

    msg = msg.format(question=question, choice_a=options[0], choice_b=options[1], choice_c=options[2], choice_d=options[3], choice_e=options[4])

    return msg

def extract_after_final_answer(text):
    """
    Extract text that comes after '**FinalAnswer**' marker
    
    Args:
        text (str): Input string containing the marker
        
    Returns:
        str: Text after the marker, or empty string if marker not found
    """
    marker = "FinalAnswer"
    
    # Find the position of the marker
    marker_pos = text.find(marker)
    
    if marker_pos == -1:
        return ""  # Marker not found
    
    # Extract everything after the marker
    start_pos = marker_pos + len(marker)
    result = text[start_pos:]
    
    return result