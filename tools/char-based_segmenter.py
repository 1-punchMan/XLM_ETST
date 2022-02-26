import sys

for line in sys.stdin:
    segmented = ' '.join(list(line.rstrip()))
    print(segmented)