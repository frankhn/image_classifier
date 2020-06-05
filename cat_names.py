import json

def cat_names(filename):
    with open(filename) as f:
        names = json.load(f)
    return names