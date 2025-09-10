import os
import numpy as np
import pickle
import random

# Define the tools and their example usage
TOOLS = {
    "CALCULATE": {
        "questions_2_args": [
            "What is {} + {}?",
            "Calculate {} times {}.",
            "Solve {} - {}.",
            "What is {} divided by {}?",
        ],
        "questions_3_args": [
            "Compute {} * {} + {}.",
        ],
        "answers": [
            "The answer is [TOOL:CALCULATE({})]",
            "It's [TOOL:CALCULATE({})]",
            "Result: [TOOL:CALCULATE({})]",
        ],
        "args_range": (1, 100),
    },
    "GET_DATE": {
        "questions": [
            "What is today's date?",
            "Tell me the current date.",
            "Date today?",
        ],
        "answers": [
            "Today's date is [TOOL:GET_DATE()]",
            "The current date is [TOOL:GET_DATE()]",
            "It's [TOOL:GET_DATE()]",
        ],
        "args_range": None,
        "num_args": [0, 0, 0],
    },
    "SEARCH": {
        "questions": [
            "Who is the president of {}?",
            "What is the capital of {}?",
            "Tell me about {}.",
            "Search for {}.",
        ],
        "answers": [
            "Let me search. [TOOL:SEARCH({})]",
            "I can find that. [TOOL:SEARCH({})]",
            "Searching for [TOOL:SEARCH({})]",
        ],
        "args_list": [
            "USA", "France", "Japan", "Germany", "Canada", "Australia",
            "Artificial Intelligence", "Quantum Physics", "Machine Learning",
            "Mount Everest", "Great Wall of China",
        ],
        "num_args": [1, 1, 1, 1],
    },
}

def generate_example(tool_name):
    tool_info = TOOLS[tool_name]
    answer_template = random.choice(tool_info["answers"])

    if tool_name == "CALCULATE":
        # Randomly choose between 2-arg and 3-arg questions
        if random.random() < 0.8: # Bias towards 2-arg questions
            question_templates = tool_info["questions_2_args"]
            num_args_for_template = 2
        else:
            question_templates = tool_info["questions_3_args"]
            num_args_for_template = 3

        question_template = random.choice(question_templates)
        args_for_template = [random.randint(*tool_info["args_range"]) for _ in range(num_args_for_template)]

        # Determine the operation and build the expression string
        if num_args_for_template == 2:
            if "+" in question_template: op = "+"
            elif "-" in question_template: op = "-"
            elif "times" in question_template or "*" in question_template: op = "*"
            elif "divided by" in question_template or "/" in question_template: op = "/"
            expression = f"{args_for_template[0]}{op}{args_for_template[1]}"
        elif num_args_for_template == 3:
            expression = f"{args_for_template[0]}*{args_for_template[1]}+{args_for_template[2]}"
        
        question = question_template.format(*args_for_template)
        answer = answer_template.format(expression)
    elif tool_name == "GET_DATE" or tool_name == "SEARCH": # Handle GET_DATE and SEARCH here
        question_template = random.choice(tool_info["questions"])
        if tool_name == "GET_DATE":
            question = question_template
            answer = answer_template
        elif tool_name == "SEARCH":
            arg = random.choice(tool_info["args_list"])
            question = question_template.format(arg)
            answer = answer_template.format(arg)
    
    return f"Q: {question}\nA: {answer}<|endoftext|>"


def generate_dataset(num_examples_per_tool=1000):
    all_examples = []
    for tool_name in TOOLS:
        for _ in range(num_examples_per_tool):
            all_examples.append(generate_example(tool_name))
    random.shuffle(all_examples)
    return "".join(all_examples)

# --- Tokenization (simplified for character-level) ---
# This part assumes a character-level tokenizer similar to tinystories
# For a real GPT model, you'd use a BPE tokenizer like tiktoken

def get_tokenizer_info(data_dir):
    meta_path = os.path.join(data_dir, 'meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        return meta['stoi'], meta['itos'], meta['vocab_size']
    else:
        # Fallback if meta.pkl not found (e.g., for initial run)
        # This should ideally match the model's tokenizer
        chars_list = [
            "Q:A: ",
            "[TOOL:CALCULATE()]",
            "+-*/0123456789",
            "[TOOL:GET_DATE()]",
            "[TOOL:SEARCH()]",
            "abcdefghijklmnopqrstuvwxyz",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            ".,?!'\"{}[]()<>|",
            "\n", # Newline character
        ]
        chars_str = "".join(chars_list)
        chars = sorted(list(set(chars_str)))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}
        vocab_size = len(chars)
        return stoi, itos, vocab_size

def encode(s, stoi):
    return [stoi[c] for c in s]

# --- Main script ---
if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__))
    
    # Get tokenizer info (assuming it's consistent with the model's tokenizer)
    # For this synthetic data, we'll create a simple character-level tokenizer
    # that includes all characters needed for tool calls.
    stoi, itos, vocab_size = get_tokenizer_info(os.path.join(os.path.dirname(__file__), '..', 'tinystories'))
    
    print(f"Generating dataset with vocab size: {vocab_size}")

    full_text = generate_dataset(num_examples_per_tool=500) # Generate 500 examples per tool
    
    # Split into train and val
    n = len(full_text)
    train_text = full_text[:int(n*0.9)]
    val_text = full_text[int(n*0.9):]

    train_ids = encode(train_text, stoi)
    val_ids = encode(val_text, stoi)

    print(f"Train has {len(train_ids):,} tokens")
    print(f"Val has {len(val_ids):,} tokens")

    # Export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(data_dir, 'train.bin'))
    val_ids.tofile(os.path.join(data_dir, 'val.bin'))

    # Save meta information
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)

    print("Dataset generation complete.")
