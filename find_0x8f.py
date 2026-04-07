import os

for root, dirs, files in os.walk('.'):
    for name in files:
        if name.endswith('.pyc') or name.startswith('.') or 'venv' in root:
            continue
        path = os.path.join(root, name)
        try:
            with open(path, 'rb') as f:
                data = f.read()
                if b'\x8f' in data:
                    pos = data.find(b'\x8f')
                    print(f"Found 0x8f in {path} at position {pos}")
        except Exception as e:
            pass
