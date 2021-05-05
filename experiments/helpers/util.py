import json
from datetime import datetime
import pytz
import os

my_timezone = os.getenv('PY_TZ', 'America/Toronto')

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
        
def get_time_with_tz():
    return datetime.now().astimezone(pytz.timezone(my_timezone))
