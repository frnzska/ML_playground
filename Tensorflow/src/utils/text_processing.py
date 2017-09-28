from typing import List, Dict
import re

def text_to_idx_dict(*, text:str) -> Dict:
    """
    Transforms a text sequence to dictionary with index representation of word. 
    Index from alphabetical ordering
    :param text: text to be transformed
    :return: dictionary (inf)
    """
    words = re.findall(r'\w+', text.lower())
    words = sorted(list(set(words)))
    return {w:words.index(w) for w in words}
