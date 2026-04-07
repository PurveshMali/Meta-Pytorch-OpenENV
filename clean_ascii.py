import os
import re

def clean_to_ascii(text):
    # Map some common symbols
    replacements = {
        '—': '-',
        '→': '->',
        '📧': '[Email Triage]',
        '🎯': '[Goal]',
        '🗂️': '[Structure]',
        '🎮': '[Action Space]',
        '👁️': '[Observation]',
        '📋': '[Tasks]',
        '🏆': '[Rewards]',
        '⚙️': '[Setup]',
        '📊': '[Baseline]',
        '🌐': '[Endpoints]',
        '🧪': '[Testing]',
        '📄': '[Environment]',
        '✓': '[YES]',
        '⚠': '[WARNING]',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # Remove any remaining non-ASCII
    return text.encode('ascii', 'ignore').decode('ascii')

files_to_clean = ['README.md', 'openenv.yaml']

for f_path in files_to_clean:
    if os.path.exists(f_path):
        with open(f_path, 'r', encoding='utf-8') as f:
            content = f.read()
        cleaned = clean_to_ascii(content)
        with open(f_path, 'w', encoding='ascii') as f:
            f.write(cleaned)
        print(f"Cleaned {f_path}")
