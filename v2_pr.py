import spacy
import re

nlp = spacy.load("en_core_web_md")

def hide_sensitive_information(source_file, destination_file):
    with open(source_file, 'r') as f:
        content = f.read()

    doc = nlp(content)
    for entity in doc.ents:
        if entity.label_ in ['PERSON', 'GPE', 'DATE', 'ORG']:
            content = content.replace(entity.text, '██████')
    
    content = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", '██████', content)
    content = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", '██████', content)
    content = re.sub(r"\bmedical record number is \d+\b", '██████', content, flags=re.IGNORECASE)
    content = re.sub(r"\b\d{5,}\b", '██████', content)
    content = re.sub(r"\b(\d{3}[-.]?\d{4}|\d{3}[-.]?\d{3}[-.]?\d{4})\b", '██████', content)

    with open(destination_file, 'w') as f:
        f.write(content)

source_file = '/Users/rohit1208/Desktop/Med-Kick Project/transcript.txt'
destination_file = '/Users/rohit1208/Desktop/Med-Kick Project/output.txt'
hide_sensitive_information(source_file, destination_file)
