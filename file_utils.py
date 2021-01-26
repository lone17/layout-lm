import os, json
from typing import *
import re
from reader.file_reader.file_reader import *
from tqdm import tqdm
import glob, tqdm

sep = ' . '


def read_document(file):
    file = FileReader(file)
    json.dump(file.to_dict(), open('test_.json', 'w', encoding='utf-8'), ensure_ascii=False)
    return file.to_dict()


# convert pdf to json
def read_data_from_pdf_to_json(pdf_folder, output_folder):
    import glob
    number_error = 0
    errors = []
    files = glob.glob(pdf_folder + '/*.pdf')

    for i, file in enumerate(files):
        try:
            file_name = re.split(r'\\|\/', file)[-1]
            print(f"Read file {i}: {file}")
            data = FileReader(file)
            with open(output_folder + '/' + file_name + '.json', 'w', encoding='utf-8') as f:
                json_data = data.to_dict()
                json.dump(json_data, f, ensure_ascii=False)
        except:
            errors.append(file)
    print(f"PROCESSED {len(files)} files with {len(errors)} ERRORS!")
    with open("number_erroro.txt", 'a') as f:
        f.write(pdf_folder + " error " + str(number_error) + '\n')
    with open("list_error.json", 'w', encoding='utf-8') as f:
        json.dump(errors, f)
    print("DONE!")


def convert_data_to_layout_lm(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    pages_data = []
    sep = ' '
    for page in data:

        word_label = {'words': [], 'labels': [], 'size': [page['height'], page['width']]}
        for line in page['textlines']:
            words = line['words']
            all_start_positions = [w['start_position'] for w in words]
            answers = line['answers']
            all_answer_word_positions = {}
            answers = sorted(answers, key=lambda x: x['start_pos'])
            for answer in answers:
                label = answer['label']
                all_answer_word_positions[answer['start_pos']] = label
                text = answer['text']
                for i in range(len(text)):
                    if text[i] == sep:
                        all_answer_word_positions[i + 1 + answer['start_pos']] = label
            labels = []
            label = 'O'
            for i in range(len(words)):
                if all_start_positions[i] in all_answer_word_positions:
                    current_label = all_answer_word_positions[all_start_positions[i]]
                    if current_label == label:
                        labels.append('I-' + current_label)
                    else:
                        labels.append('B-' + current_label)
                        label = current_label
                else:
                    labels.append('O')
                    label = 'O'
            assert len(words) == len(labels)
            word_label['words'].append(words)
            word_label['labels'].append(labels)
        pages_data.append(word_label)
    return pages_data


def write_data(pages_data, output_folder, mode='train'):
    train_file = output_folder + f'/{mode}.txt'
    train_box_file = output_folder + f'/{mode}_box.txt'
    img_file = output_folder + f'/{mode}_image.txt'
    trainf = open(train_file, 'w', encoding='utf-8')
    boxf = open(train_box_file, 'w', encoding='utf-8')
    imgf = open(img_file, 'w', encoding='utf-8')

    for page in pages_data:
        size = page['size']
        height = size[0]
        width = size[1]
        assert len(page['words']) == len(page['labels'])
        words_page = [w for words in page['words'] for w in words]
        labels_page = [label for labels in page['labels'] for label in labels]
        for i in range(len(words_page)):
            text = words_page[i]['text'].replace('\n', '')
            if text == 'App\n':
                1/0
            label = labels_page[i]
            x0 = words_page[i]['x0']
            x0_normalized = int(x0 / width * 1000)
            x1 = words_page[i]['x1']
            x1_normalized = int(x1 / width * 1000)
            y0 = words_page[i]['y0']
            y0_normalized = int(y0 / height * 1000)
            y1 = words_page[i]['y1']
            y1_normalized = int(y1 / height * 1000)
            if i % 400 == 0:
                trainf.write('\n')
                boxf.write('\n')
                imgf.write('\n')
            trainf.write(f"{text}\t{label}\n")
            boxf.write(f"{text}\t{x0_normalized} {y0_normalized} {x1_normalized} {y1_normalized}\n")
            imgf.write(f"{text}\t{x0} {y0} {x1} {y1}\t{width} {height}\tnone\n")


if __name__ == "__main__":
    # TODO: CONVERT PDF TO json
    # read_data_from_pdf_to_json('E:\\CV\\ihr_production',#Folder chứa file được dán nhãn
    #                            'E:\\CV\\json_data')#Folder đích chứa file json
    # # TODO: convert from json folder to conll
    # convert_data_to_colln(data_folder='C:\\Users\Levi\Desktop\\test__',#Folder chứa file json
    #                       output_folder='C:\\Users\Levi\Desktop\\annotation_reader', mode='train')#Train, test, dev
    folder = 'E:\\CV\\json_data'
    output_folder_train = 'D:\\layout-lm\\data/train'
    output_folder_test = 'D:\\layout-lm\\data/test'
    files = glob.glob(folder + '/*.json')
    pages_data = []
    print("write train data ...")

    error_train = 0
    error_test = 0
    for file in tqdm.tqdm(files[:int(0.9 * len(files))]):
        try:
            pages_data.extend(convert_data_to_layout_lm(file))
        except:
            error_train += 1
    write_data(pages_data, output_folder_train)
    print(f"got {error_train} errors in train data")
    print("write test data ...")
    pages_data = []
    for file in tqdm.tqdm(files[int(0.9 * len(files)):]):
        try:
            pages_data.extend(convert_data_to_layout_lm(file))
        except:
            error_test += 1
    write_data(pages_data, output_folder_test, mode='test')
    print(f"got {error_test} errors in test data")

    # file = 'C:\\Users\\cinnamon\\Desktop/multi-answer.pdf'
    # read_document(file)
