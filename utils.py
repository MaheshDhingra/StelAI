import json

def save_config(config, filepath):
    """Saves a configuration dictionary to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(filepath):
    """Loads a configuration dictionary from a JSON file."""
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config
