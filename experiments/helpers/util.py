import json

def save_json_file(data, filename):
    with open(filename, 'w') as outfile:
        json.dump(data, outfile)

def load_json_file(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
    return data

def zero_if_none(inp):
    if inp is None:
        return 0
    else:
        return inp
        