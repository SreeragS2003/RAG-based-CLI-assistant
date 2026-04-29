import re

def strip_markdown(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)        # italic
    text = re.sub(r'^- ', '', text, flags=re.MULTILINE)  # bullets
    return text.strip()