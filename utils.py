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


def bbox_string(box, width, length):
    return (
        str(int(1000 * (box[0] / width)))
        + " "
        + str(int(1000 * (box[1] / length)))
        + " "
        + str(int(1000 * (box[2] / width)))
        + " "
        + str(int(1000 * (box[3] / length)))
    )


def actual_bbox_string(box, width, length):
    return (
        str(box[0])
        + " "
        + str(box[1])
        + " "
        + str(box[2])
        + " "
        + str(box[3])
        + "\t"
        + str(width)
        + " "
        + str(length)
    )


def sort_funsd_reading_order(lines):
    """ Sort cell list to create the right reading order using their locations
    Parameters
    ----------
    lines: list of cells to sort
    Returns
    -------
    a list of cell lists in the right reading order that contain no key or start with a key and contain no other key
    """
    sorted_list = []
    
    if len(lines) == 0:
        return lines
    
    while len(lines) > 1:
        topleft_line = lines[0]
        for line in lines[1:]:
            topleft_line_pos = topleft_line['box']
            topleft_line_center_y = (topleft_line_pos[1] + 
                                     topleft_line_pos[3]) / 2
            x1, y1, x2, y2 = line['box']
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            cell_h = y2 - y1
            if box_center_y <= topleft_line_center_y - cell_h / 2:
                topleft_line = line
                continue
            if box_center_x < topleft_line_pos[2] and box_center_y < topleft_line_pos[3]:
                topleft_line = line
                continue
        sorted_list.append(topleft_line)
        lines.remove(topleft_line)
    
    sorted_list.append(lines[0])
    
    return sorted_list


# print(convert_SO_to_BIOES(['O'] * 5 + ['S-x'] * 4 + ['O'] * 3 + ['S-y'] * 2 + ['O'] * 5 + ['S-z'] * 1
#                           + ['S-t'] * 3 + ['S-u'] * 4 + ['O'] + ['S-w'] * 3))