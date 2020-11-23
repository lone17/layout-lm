import argparse
import json
import os
from transformers import AutoTokenizer
from fuzzysearch import find_near_matches
from fuzzywuzzy import fuzz
import openpyxl
import glob
from collections import Counter


def bbox_string(box, width, length):
    """
    Get normalized bounding-box string in [0 -1000] for LayoutLM model
    Args:
        box: a box with 4 components [x1, y1, x2, y2]
        width: width of the input image
        length: height of the input image

    Returns:
        Normalize bounding-box string
    """
    assert box[0] <= width and box[2] <= width
    assert box[1] <= length and box[3] <= length

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
    """
    Get actual bounding-box string
    Args:
        box: a box with 4 components [x1, y1, x2, y2]
        width: width of the input image
        length: height of the input image

    Returns:
        Normalize bounding-box string
    """
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


def get_max_shape(json_toshiba):
    """
    Get maximum shape of text-boxes (approximate the page size) from input list
    Args:
        json_toshiba: JSON dict format of Toshiba dataset (list of text-boxes)

    Returns:
        Maximum dimension (x, y)

    """
    max_x = max_y = 0.0
    for item in json_toshiba:
        if 'x1' in item and float(item['x1']) > max_x:
            max_x = float(item['x1'])
        if 'bottom' in item and float(item['bottom']) > max_y:
            max_y = float(item['bottom'])

    return max_x, max_y


def split_json_by_page(json_toshiba):
    """
    Split the input JSON and group by page using heuristics
    Args:
        json_toshiba: JSON dict format of Toshiba dataset (list of text-boxes)

    Returns:
        list of item in JSON grouped by page

    """
    current_y_pos = 0
    list_pages = []
    current_page = []
    N = len(json_toshiba)
    for idx, item in enumerate(json_toshiba):
        # if the box doesn't contain positional information, skip (table)
        if 'x1' not in item or 'top' not in item:
            continue
        x0, x1, y0, y1 = [float(item[k]) for k in ['x0', 'x1', 'top', 'bottom']]
        if y1 < current_y_pos:
            list_pages.append(current_page)
            current_page = [item]
        else:
            current_page.append(item)
        current_y_pos = y1
        if idx == N - 1:
            list_pages.append(current_page)
    return list_pages


def read_excel_ca_single(ca_path):
    """
    Read CA for NLP data from Excel format
    Args:
        ca_path: path to CA file

    Returns:
        List of text lines in the CA

    """
    # Give the location of the file
    print('Reading CA', ca_path)
    ca_lines = []

    wb_obj = openpyxl.load_workbook(ca_path)
    sheet = wb_obj.active
    idx = 0

    for row in sheet.iter_rows():
        if idx == 0:
            idx += 1
            continue
        idx += 1
        values = [row[k].value for k in range(3, 7)]
        ca_lines.append(values)

    return ca_lines


def group_ca_by_block(ca_lines):
    """
    Group CA lines by paragraph blocks (same parent ID)
    Args:
        ca_lines:

    Returns:
        List of indexes which separation start
    """
    current_parent_idx = ca_lines[0][-1]
    sep_indexes = []

    for idx, line in enumerate(ca_lines):
        text, title, index, parent_index = line
        if parent_index != current_parent_idx and parent_index == index:
            sep_indexes.append(idx - 1)
            # sep_indexes.append(idx)
            current_parent_idx = parent_index

    return sep_indexes


def get_title_ca_block(ca_lines):
    """
    Get title lines from CA (line which has child based on parent idx)
    Args:
        ca_lines:

    Returns:

    """
    title_indexes = []

    for idx, line in enumerate(ca_lines):
        text, title, index, parent_index = line
        if idx == len(ca_lines) - 1:
            continue
        if index == ca_lines[idx + 1][3]:
            title_indexes.append((idx, 1))
            # title_indexes.append((idx - 1, 0))

    return title_indexes


def read_excel_ca_from_path(ca_dir):
    """
    Read CA from folder
    Args:
        ca_dir:

    Returns:

    """
    ca_dict = {}
    for file in glob.iglob(ca_dir + "/*.xlsx", recursive=True):
        file_name = os.path.basename(file)[:5]
        ca_dict[file_name] = read_excel_ca_single(file)

    return ca_dict


def convert_position(contexts, pos):
    offset = 1
    while len(' '.join(contexts[:offset])) < pos - 1:
        offset += 1
        if offset > len(contexts):
            break

    return offset - 1, pos - len(' '.join(contexts[:offset - 1])) - 1


def post_process_tag(output_pages):
    """
    Post-process the output of pages to add missing BEGIN or END tag
    Args:
        output_pages: list of pages in the input document: pages -> blocks -> words

    Returns:
        Same output_pages with processed tag

    """
    # one pass to detect the begin tag for each end tag detected
    for page in output_pages:
        for block in page:
            for w_idx, word in enumerate(block):
                if word['status'] == 'end':
                    start_idx = w_idx - 1
                    while not(start_idx <= 0 or block[start_idx]['status'] == 'start' or (
                            start_idx > 0 and block[start_idx - 1]['status'] == 'end')):
                        block[start_idx]['status'] = 'inside'
                        start_idx -= 1
                    if start_idx >= 0:
                        block[start_idx]['status'] = 'begin'

                if word['status'] == 'end_block':
                    start_idx = w_idx - 1
                    while not(start_idx <= 0 or (
                            start_idx > 0 and block[start_idx - 1]['status'] == 'end')):
                        start_idx -= 1
                    if start_idx >= 0:
                        block[start_idx]['status'] = 'begin'
                        if start_idx > 0:
                            block[start_idx - 1]['status'] = 'end'

            if len(block) == 1 and block[0]['status'] == 'end':
                block[0]['status'] = 'single'

    # another pass to fill the missing blocks between tags
    # next_word_begin = False
    # for page in output_pages:
    #     for b_idx, block in enumerate(page):
    #         for w_idx, word in enumerate(block):
    #             if next_word_begin:
    #                 word['status'] = 'begin'
    #                 next_word_begin = False
    #             elif word['status'] == 'begin':
    #                 if w_idx > 0:
    #                     block[w_idx - 1]['status'] = 'end'
    #                 elif b_idx > 0 and len(page[b_idx - 1]) > 0:
    #                     page[b_idx - 1][-1]['status'] = 'end'
    #             elif word['status'] == 'end':
    #                 next_word_begin = True

    word_status_list = []
    for page in output_pages:
        for block in page:
            for w_idx, word in enumerate(block):
                word_status_list.append(word['status'])
    tag_count = Counter(word_status_list)
    print('tag count', tag_count)

    return output_pages


def convert_to_funsd_format(json_toshiba, end_tag_indexes):
    """
    Convert JSON Toshiba to FUNSD format: pages -> blocks -> words
    Args:
        json_toshiba:
        end_tag_indexes: index for the block separations

    Returns:

    """
    list_pages = split_json_by_page(json_toshiba)
    output_pages = []
    line_idx = 0
    end_tag_dict = {}

    for k, v in end_tag_indexes:
        if k in end_tag_dict:
            end_tag_dict[k].append(v)
        else:
            end_tag_dict[k] = [v]

    for page in list_pages:
        json_funsd = []

        current_parent_idx = None
        current_block = []

        for item in page:
            line_idx += 1
            # if the box doesn't contain positional information, skip (table)
            if 'x1' not in item or 'top' not in item:
                continue

            x0, x1, y0, y1 = [float(item[k]) for k in ['x0', 'x1', 'top', 'bottom']]
            text = item['text']
            text = preprocess_text(text)

            words = text.split(' ')

            parent_index = item['parent_index']

            num_lines = max(int((y1 - y0) / AVG_LINE_HEIGHT), 1)
            num_words = len(words)

            if line_idx - 1 in end_tag_dict:
                end_word_indexes = [convert_position(words, offset)[0] for offset in end_tag_dict[line_idx - 1]]
            else:
                end_word_indexes = []

            words_per_line = max(num_words // num_lines + 1, 1)
            base_x = (x1 - x0) / words_per_line
            base_y = (y1 - y0) / num_lines

            # skip if word length is too long
            if base_x > 200:
                continue

            funsd_words = []
            # funsd_words.append({
            #             "box": [x0, y0, x1, y1],
            #             "text": text
            #         })

            for index, word in enumerate(words):
                col_offset = index % words_per_line
                row_offset = index // words_per_line
                assert row_offset < num_lines, "{} {} {} {} {}".format(row_offset, num_lines, index, words_per_line, num_words)
                start_x = x0 + col_offset * base_x
                start_y = y0 + row_offset * base_y
                assert start_x + base_x <= x1 + 10 and start_y + base_y <= y1 + 10, \
                    "num line {} num word {}   {} {} {} {} {} {}".format(num_lines, num_words, start_x, base_x, start_y, base_y, x1, y1)
                if len(word.strip()) > 0:
                    funsd_words.append({
                        "box": [start_x, start_y, start_x + base_x, start_y + base_y],
                        "text": word,
                        "status": "end" if index in end_word_indexes else 'start' if index == 0 else 'other'
                    })

            # continuously add words to the same block with same parent idx
            if parent_index is not None and parent_index == current_parent_idx:
                current_block += funsd_words
            else:
                if len(current_block) > 1:
                    current_block[0]['status'] = 'start'
                    current_block[-1]['status'] = 'end'
                json_funsd.append(current_block)
                current_block = funsd_words
                current_parent_idx = parent_index

        if len(current_block) > 0:
            if len(current_block) > 1:
                current_block[0]['status'] = 'start'
                current_block[-1]['status'] = 'end'
            json_funsd.append(current_block)

        output_pages.append(json_funsd)

    output_pages = post_process_tag(output_pages)

    return output_pages


def preprocess_text(line):
    return line.replace('..', '').replace('/t', ' ').replace('  ', ' ')


def preprocess_lines(lines):
    return [preprocess_text(l).lower() for l in lines]


def find_end_tags(label_lines, ca_lines, separate_indexes):
    """
    Map the label JSON with CA from Excel files to find title / paragraph separator position
    Args:
        label_lines: input JSON
        ca_lines:  CA lines
        separate_indexes: indexes of paragraph separator in ca_lines

    Returns:
        List of separate indexes for label_lines
         [(label_idx, offset_idx)]

    """
    current_label_idx = 0
    end_indexes = []
    end_type_list = []

    last_matched_label_idx = -1

    N = len(separate_indexes)
    init_fail_thres = 150
    fail_thres = init_fail_thres

    for _id, sep_idx in enumerate(separate_indexes):
        sep_idx, sep_type = sep_idx
        query_end_text = ca_lines[sep_idx]
        if sep_idx < len(ca_lines) - 1:
            query_begin_text = ca_lines[sep_idx+1]
        else:
            continue
        print("\nFinding context for ...  {} / {} '{}  |  {}' \n ".format(_id+1, N, query_end_text, query_begin_text))

        while True:
            if current_label_idx >= len(label_lines):
                break
            context_list = label_lines[current_label_idx:current_label_idx+3]
            current_context = ' '.join(context_list)

            query_end_text = query_end_text[-100:]
            query_begin_text = query_begin_text[:100]
            current_label_line = label_lines[current_label_idx]

            # if (len(query_end_text) > 20 and abs(
            #         len(query_end_text) - len(current_label_line)) < 6) and fuzz.partial_ratio(
            #         query_end_text, current_label_line) > 80 \
            #         and 0.3 < len(query_end_text) / len(current_label_line) < 1.5\
            #         and min(len(query_end_text), len(current_label_line)) > 30:
            #     end_indexes.append((current_label_idx, len(current_label_line)))
            #     end_type_list.append(sep_type)
            #     print('Match #1', current_label_line)
            #     last_matched_label_idx = current_label_idx
            #     fail_thres = init_fail_thres
            #     break
            #
            # if (len(query_begin_text) > 20 and abs(
            #         len(query_begin_text) - len(current_label_line)) < 6) and fuzz.partial_ratio(
            #         query_begin_text, current_label_line) > 80\
            #         and 0.3 < len(query_begin_text) / len(current_label_line) < 1.5\
            #         and min(len(query_end_text), len(current_label_line)) > 30:
            #     end_indexes.append((current_label_idx - 1, len(label_lines[current_label_idx - 1])))
            #     end_type_list.append(sep_type)
            #     print('Match #2', label_lines[current_label_idx - 1])
            #     last_matched_label_idx = current_label_idx
            #     fail_thres = init_fail_thres
            #     break

            search_result_end = find_near_matches(query_end_text, current_context,
                                                  max_l_dist=max(min(int(len(query_end_text) * 0.2) if len(
                                                      query_end_text) > 25 else 3, 10), 2))
            search_result_begin = find_near_matches(query_begin_text, current_context,
                                                    max_l_dist=max(min(int(len(query_begin_text) * 0.2 if len(
                                                        query_begin_text) > 25 else 3), 10), 2))

            if len(search_result_end) > 0 and len(search_result_begin) > 0:
                end_tag_idx = 0
                while end_tag_idx < len(search_result_end) and search_result_end[end_tag_idx].end <= search_result_begin[0].start:
                    end_tag_idx += 1

                offset, pos = convert_position(context_list, search_result_end[end_tag_idx-1].end)
                end_indexes.append((current_label_idx + offset, pos))
                end_type_list.append(sep_type)
                print('Match #3', label_lines[current_label_idx + offset][:pos],
                      "  |  {}  |  {}  ".format(search_result_end[0].matched, search_result_begin[0].matched))
                print(label_lines[current_label_idx + offset - 1])
                print()
                print(label_lines[current_label_idx + offset])
                last_matched_label_idx = current_label_idx
                fail_thres = init_fail_thres
                break
            current_label_idx += 1

            # skip current
            if current_label_idx - last_matched_label_idx > fail_thres:
                print('FAILED')
                # exit()
                current_label_idx = last_matched_label_idx
                fail_thres += init_fail_thres
                break

            if current_label_idx > len(label_lines) - 4:
                break

        if len(end_indexes) > 0:
            line_idx, offset = end_indexes[-1]
            print('VERIFY', line_idx, offset, label_lines[line_idx][offset - 50:offset])

    if len(end_indexes) < len(separate_indexes):
        print('FAILED')
        print(len(end_indexes), len(separate_indexes))

    return end_indexes, end_type_list


def convert(args):
    """
    Main convert function to write the input JSON to LayoutLM text file format
    Args:
        args:
    """
    if args.so_only:
        b_tag = 'S-'
        i_tag = 'S-'
        e_tag = 'S-'
    else:
        b_tag = 'B-'
        i_tag = 'I-'
        e_tag = 'E-'

    ca_dict = read_excel_ca_from_path(args.ca_dir)

    with open(
        os.path.join(args.output_dir, args.data_split + ".txt.tmp"),
        "w",
        encoding="utf8",
    ) as fw, open(
        os.path.join(args.output_dir, args.data_split + "_box.txt.tmp"),
        "w",
        encoding="utf8",
    ) as fbw, open(
        os.path.join(args.output_dir, args.data_split + "_image.txt.tmp"),
        "w",
        encoding="utf8",
    ) as fiw:
        for file in os.listdir(args.data_dir):
            file_path = os.path.join(args.data_dir, file)
            file_name = os.path.basename(file_path)[:5]
            if not file.endswith('json'):
                continue
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)

            data = [item for item in data if 'Commented' not in item['text']]

            print('PROCESSING ...', file)

            label_lines = [l['text'] for l in data]
            label_lines = preprocess_lines(label_lines)
            # separate_index = group_ca_by_block(ca_dict[file_name])
            title_indexes = get_title_ca_block(ca_dict[file_name])
            ca_lines = [c[0] for c in ca_dict[file_name]]
            ca_lines = preprocess_lines(ca_lines)

            end_tag_indexes, end_type_list = find_end_tags(label_lines,
                                                           ca_lines, title_indexes)
            for label_idx, offset in end_tag_indexes:
                print(label_idx, offset, label_lines[label_idx][offset - 50:offset])

            data = convert_to_funsd_format(data, end_tag_indexes)

            next_is_end = False
            end_tag_idx = 0

            for page_idx, page in enumerate(data):
                for block in page:
                    label = 'block'
                    file_name = file + '_page_' + str(page_idx)
                    words = block
                    if len(words) == 0:
                        continue
                    if len(words) == 1:
                        fw.write(words[0]["text"] + "\tS-" + label.upper() + "\n")
                        fbw.write(
                            words[0]["text"]
                            + "\t"
                            + bbox_string(words[0]["box"], width, length)
                            + "\n"
                        )
                        fiw.write(
                            words[0]["text"]
                            + "\t"
                            + actual_bbox_string(words[0]["box"], width, length)
                            + "\t"
                            + file_name
                            + "\n"
                        )
                    else:
                        # fw.write(words[0]["text"] + "\t" + b_tag + label.upper() + "\n")
                        # fbw.write(
                        #     words[0]["text"]
                        #     + "\t"
                        #     + bbox_string(words[0]["box"], width, length)
                        #     + "\n"
                        # )
                        # fiw.write(
                        #     words[0]["text"]
                        #     + "\t"
                        #     + actual_bbox_string(words[0]["box"], width, length)
                        #     + "\t"
                        #     + file_name
                        #     + "\n"
                        # )
                        for w_idx, w in enumerate(words): #words[1:-1]):
                            if w["status"] == "end":
                                # print('End tag')
                                fw.write(w["text"] + "\t" + e_tag + label.upper() + "\n")
                                next_is_end = False
                            elif next_is_end:
                                fw.write(w["text"] + "\t" + b_tag + label.upper() + "\n")
                                next_is_end = False
                            elif w["status"] == "begin":
                                fw.write(w["text"] + "\t" + b_tag + label.upper() + "\n")
                            # elif w["status"] == "inside":
                            #     # print('Inside tag')
                            #     fw.write(w["text"] + "\t" + i_tag + label.upper() + "\n")
                            else:
                                fw.write(w["text"] + "\t" + i_tag + label.upper() + "\n")
                            fbw.write(
                                w["text"]
                                + "\t"
                                + bbox_string(w["box"], width, length)
                                + "\n"
                            )
                            fiw.write(
                                w["text"]
                                + "\t"
                                + actual_bbox_string(w["box"], width, length)
                                + "\t"
                                + file_name
                                + "\n"
                                )
                        # fw.write(words[-1]["text"] + "\t" + e_tag + label.upper() + "\n")
                        # fbw.write(
                        #     words[-1]["text"]
                        #     + "\t"
                        #     + bbox_string(words[-1]["box"], width, length)
                        #     + "\n"
                        # )
                        # fiw.write(
                        #     words[-1]["text"]
                        #     + "\t"
                        #     + actual_bbox_string(words[-1]["box"], width, length)
                        #     + "\t"
                        #     + file_name
                        #     + "\n"
                        # )
                fw.write("\n")
                fbw.write("\n")
                fiw.write("\n")


def seg_file(file_path, tokenizer, max_len, enforce_end_tag=False, separate_indexes=None):
    subword_len_counter = 0
    new_separate_indexes = []
    remove_indexes = []
    output_path = file_path[:-4]

    if separate_indexes is not None:
        separate_indexes, to_remove_indexes = separate_indexes
    else:
        to_remove_indexes = []

    with open(file_path, "r", encoding="utf8") as f_p, open(
        output_path, "w", encoding="utf8"
    ) as fw_p:
        last_end_idx = 0

        for idx, line in enumerate(f_p):
            line = line.rstrip()
            item = line.split()

            if not line:
                if separate_indexes is not None and idx > separate_indexes[-1]:
                    continue
                fw_p.write(line + "\n")
                subword_len_counter = 0
                continue

            if enforce_end_tag:
                tag = item[-1][0]

            if separate_indexes is None:
                token = line.split("\t")[0]

                current_subwords_len = len(tokenizer.tokenize(token))

                # Token contains strange control characters like \x96 or \x95
                # Just filter out the complete line
                if current_subwords_len == 0:
                    remove_indexes.append(idx)
                    continue

                if (subword_len_counter + current_subwords_len) > max_len:
                    if not enforce_end_tag:
                        fw_p.write("\n" + line + "\n")
                        subword_len_counter = current_subwords_len
                        new_separate_indexes.append(idx)
                    else:
                        if tag == "E":
                            last_end_idx = idx
                        new_separate_indexes.append(last_end_idx)
                        subword_len_counter = idx - last_end_idx + current_subwords_len
                    continue

                subword_len_counter += current_subwords_len
                fw_p.write(line + "\n")
            else:
                if idx in to_remove_indexes:
                    continue
                if idx in separate_indexes:
                    fw_p.write("\n" + line + "\n")
                else:
                    if idx <= separate_indexes[-1]:
                        fw_p.write(line + "\n")
    return new_separate_indexes, remove_indexes


def seg(args):
    """
    Separate input file input smaller block based on maximum token length
    Args:
        args:
    """
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, do_lower_case=True
    )
    print('Finding seg indexes')
    sep_indexes = seg_file(
        os.path.join(args.output_dir, args.data_split + ".txt.tmp"),
        tokenizer,
        args.max_len,
        enforce_end_tag=False
    )
    print('Seg file txt')
    seg_file(
        os.path.join(args.output_dir, args.data_split + ".txt.tmp"),
        tokenizer,
        args.max_len,
        separate_indexes=sep_indexes
    )
    print('Seg file box')
    seg_file(
        os.path.join(args.output_dir, args.data_split + "_box.txt.tmp"),
        tokenizer,
        args.max_len,
        separate_indexes=sep_indexes
    )
    print('Seg file image')
    seg_file(
        os.path.join(args.output_dir, args.data_split + "_image.txt.tmp"),
        tokenizer,
        args.max_len,
        separate_indexes=sep_indexes
    )

    if args.data_split == 'train':
        label_list = set()
        for line in open(os.path.join(args.output_dir, 'train.txt'),
                         'r', encoding='utf8').read().split('\n'):
            if not line:
                continue
            label_list.add(line.split('\t')[-1])

        label_list = sorted(list(label_list))
        with open(os.path.join(args.output_dir, 'labels.txt'), 'w', encoding='utf8') as f:
            f.write('\n'.join(label_list))


def get_max_shape_from_dir(data_path):
    """
    Get maximum shape of text-boxes for every file in the folder
    Args:
        data_path:

    Returns:

    """
    overall_max_x = overall_max_y = 0
    for file in os.listdir(data_path):
        if not file.endswith('json'):
            continue
        with open(os.path.join(data_path, file), 'r') as fi:
            json_dict = json.load(fi)
        max_x, max_y = get_max_shape(json_dict)

        print('File {} max shape (x, y)   {}   {}'.format(file, max_x, max_y))
        overall_max_x = max(max_x, overall_max_x)
        overall_max_y = max(max_y, overall_max_y)

    print('Overall:  max shape (x, y)   {}   {}'.format(overall_max_x, overall_max_y))

    return overall_max_x, overall_max_y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, default="data_toshiba_raw/.cache/val/"
    )
    parser.add_argument("--data_split", type=str, default="val")
    parser.add_argument("--output_dir", type=str, default="data_toshiba")
    parser.add_argument("--ca_dir", type=str, default="data_toshiba_raw/ca")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--max_len", type=int, default=510)
    parser.add_argument("--so_only", type=bool, default=False)
    
    args = parser.parse_args()

    AVG_LINE_HEIGHT = 15.0
    width, length = get_max_shape_from_dir(args.data_dir)
    width, length = width + 10, length + 10

    from pathlib import Path
    p = Path(args.output_dir)
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

    convert(args)
    seg(args)
