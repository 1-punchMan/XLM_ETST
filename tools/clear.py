

import re
import sys


def clear(sent):
    clean = sent.strip().lower().replace('\n', '')
    clean = re.sub(r'[|]+', '', clean)
    clean = re.sub(r'[ ]+', '', clean)  # special space

    # remove repetition
    clean = re.sub(r'[~]+', '~', clean)
    clean = re.sub(r'[。]+', '。', clean)
    clean = re.sub(r'[_]+', '_', clean)
    clean = re.sub(r'[-]+', '-', clean)
    clean = re.sub(r'[.]+', '.', clean)
    clean = re.sub(r'[…]+', '.', clean)
    clean = re.sub(r'[ ]+', ' ', clean) # space

    clean = clean.replace('１', '1')
    clean = clean.replace('２', '2')
    clean = clean.replace('３', '3')
    clean = clean.replace('４', '4')
    clean = clean.replace('５', '5')
    clean = clean.replace('６', '6')
    clean = clean.replace('７', '7')
    clean = clean.replace('８', '8')
    clean = clean.replace('９', '9')
    clean = clean.replace('０', '0')

    return clean


for line in sys.stdin:
    line = clear(line)
    print(u'%s' % line)
