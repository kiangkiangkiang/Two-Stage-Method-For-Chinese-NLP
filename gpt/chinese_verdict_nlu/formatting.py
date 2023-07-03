import json

def do_formatting(data_file="data/testset.txt",
                  save_file="data/formatted_testset.jsonl"):

    # data_file = 'data/testset.txt'
    formatted_data = []

    with open(data_file, "r", encoding="utf-8") as textfile:

        for id, example in enumerate(textfile):
            example = json.loads(example.strip())

            formatted_example = {
                "id": id + 1,
                "data": example['text_a'],
                "label": [example['choices'][label_idx] for label_idx in example["labels"]]
            }

            formatted_data.append(formatted_example)

    with open(save_file, 'w', encoding="utf-8") as f:
        for item in formatted_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')


if __name__ == '__main__':

    do_formatting()