import sys

out = ""
for line in sys.stdin:
    segmented = ' '.join(list(line.rstrip())) + '\n'
    out += segmented

print(out)