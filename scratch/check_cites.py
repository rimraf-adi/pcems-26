import re

with open('paper/norm.tex', 'r') as f:
    lines = f.readlines()

clean_lines = [l for l in lines if not l.strip().startswith('%')]
clean_content = ''.join(clean_lines)

first_seen = {}
citation_order = []
cite_pattern = re.compile(r'\\cite\{([^}]+)\}')

for match in cite_pattern.finditer(clean_content):
    keys = [k.strip() for k in match.group(1).split(',')]
    for k in keys:
        if k not in first_seen:
            first_seen[k] = len(first_seen) + 1
            citation_order.append(k)

print("--- CITATIONS FROM SECTION 3 ONWARD ---")
for line_num, line in enumerate(clean_lines, 1):
    if line_num > 180:
        matches = cite_pattern.findall(line)
        if matches:
            nums = []
            for m in matches:
                for k in m.split(','):
                    nums.append(first_seen[k.strip()])
            print(f"Line {line_num:3d} (Refs {nums}): {line.strip()[:90]}")
