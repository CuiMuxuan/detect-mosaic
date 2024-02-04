import uuid
import re

def contains_chinese(s):
    return re.search('[\u4e00-\u9fa5]', s) is not None
def get_last_part(s):
    if not s:
        return str(uuid.uuid4())
    elif '/' in s:
        return s.split('/')[-1]
    elif '\\' in s:
        return s.split('\\')[-1]
    else:
        return str(uuid.uuid4())