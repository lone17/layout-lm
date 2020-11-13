import numpy as np
from collections import Counter

def most_common(a):
    count = {}
    max_count = 0
    result = a[0]
    for i in a:
        if i not in count:
            count[i] = 1
        else:
            count[i] += 1
        if count[i] > max_count:
            max_count = count[i]
            result = i
    
    return result


def smoothen(seq, radius=2):
    if radius == 0:
        return seq
    
    result = [-1] * len(seq)
    result[:radius] = seq[:radius]
    result[-radius:] = seq[-radius:]
    
    for i in range(radius, len(seq) - radius):
        result[i] = most_common(seq[i - radius : i + radius + 1])
    
    for i in range(radius, len(result) - radius):
        result[i] = most_common(result[i - radius : i + radius + 1])
    
    return result

def convert_SO_to_BIOES(tags):
    # print(tags)
    new_tags = []
    
    i = 0
    while i < len(tags):
        if tags[i] == 'O':
            new_tags.append(tags[i])
            i += 1
        elif tags[i].startswith('S-'):
            if i + 1 == len(tags) or tags[i+1] != tags[i]:
                new_tags.append(tags[i])
                i += 1
            else:
                new_tags.append(tags[i].replace('S-', 'B-'))
                j = i + 1
                while j < len(tags) and tags[j] == tags[i]:
                    new_tags.append(tags[i].replace('S-', 'I-'))
                    j += 1
                new_tags[-1] = new_tags[-1].replace('I-', 'E-')
                i = j
            
        else:
            raise Exception('Invalid format!')
    
    return new_tags


# print(convert_SO_to_BIOES(['O'] * 5 + ['S-x'] * 4 + ['O'] * 3 + ['S-y'] * 2 + ['O'] * 5 + ['S-z'] * 1
#                           + ['S-t'] * 3 + ['S-u'] * 4 + ['O'] + ['S-w'] * 3))