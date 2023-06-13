import argparse
from paddlenlp import Taskflow
from paddlenlp.transformers import AutoTokenizer
from utils import get_template_tokens_len
import os
from paddlenlp.utils.log import logger
from tqdm import tqdm
import numpy as np
import shutil
import json
from uie_model.filter_text import filter_text

SCHEMA = ["原告年齡", "肇事過失責任比例", "受有傷害"]
DOUBLE_CHECK = True

# TODO delete do_double_check (this for debug)
missing_recode_delete_me = {entity_type: 0 for entity_type in SCHEMA}
error_recode_delete_me = missing_recode_delete_me.copy()
total_delete_me = missing_recode_delete_me.copy()


def do_double_check(uie_output, labels):
    label_counter = {entity_type: 0 for entity_type in SCHEMA}
    found_label = label_counter.copy()

    for label in labels:
        if label < 36:
            label_counter["受有傷害"] += 1
            total_delete_me["受有傷害"] += 1
        elif label >= 36 and label < 47:
            label_counter["肇事過失責任比例"] += 1
            total_delete_me["肇事過失責任比例"] += 1
        else:
            label_counter["原告年齡"] += 1
            total_delete_me["原告年齡"] += 1

    for key in uie_output[0]:
        found_label[key] += 1

    for each_schema in SCHEMA:
        if label_counter[each_schema] > 0 and found_label[each_schema] == 0:
            missing_recode_delete_me[each_schema] += 1
            logger.debug(f"UIE 沒抓出來 「{each_schema}. Total: {missing_recode_delete_me[each_schema]}")
        elif label_counter[each_schema] == 0 and found_label[each_schema] > 0:
            error_recode_delete_me[each_schema] += 1
            logger.debug(f"UIE 抓錯 「{each_schema}. Total: {error_recode_delete_me[each_schema]}")


def write_json(data, out_path):
    with open(out_path, "w", encoding="utf-8") as outfile:
        for each_data in data:
            jsonString = json.dumps(each_data, ensure_ascii=False)
            outfile.write(jsonString)
            outfile.write("\n")


if __name__ == "__main__":
    # python uie_preprocessing.py --max_seq_len 768 --threshold 0.0 --uie_model_name_or_path uie_model/model_best/ --dataset_path ./toy_data
    # python uie_preprocessing.py --max_seq_len 768 --threshold 0.0 --uie_model_name_or_path uie_model/model_best/
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        default="./data",
        type=str,
        help="Local dataset directory including train.txt, dev.txt and label.txt (optional).",
    )
    parser.add_argument("--train_file", type=str, default="train.txt", help="Train dataset file name.")
    parser.add_argument("--dev_file", type=str, default="dev.txt", help="Dev dataset file name.")
    parser.add_argument("--test_file", type=str, default="test.txt", help="Test dataset file name.")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Max length of sequence.")
    parser.add_argument("--special_word_len", type=int, default=200, help="Length of special word.")
    parser.add_argument(
        "--threshold", type=float, default=0.4, help="Filter threshold (probability < threshold will be remove)"
    )
    parser.add_argument(
        "--uie_model_name_or_path",
        type=str,
        default="uie-base",
        help="The build-in pretrained UIE model name or path to its checkpoints, such as "
        "`uie-base`, `uie-medium`, `uie-mini`.",
    )
    parser.add_argument(
        "--utc_model_name_or_path",
        type=str,
        default="utc-base",
        help="The build-in pretrained UTC model name or path to its checkpoints, such as "
        "`utc-xbase`, `utc-base`, `utc-medium`, `utc-mini`, `utc-micro`, `utc-nano` and `utc-pico`.",
    )
    parser.add_argument("--dynamic_adjust_length", type=str, default="True", help="aaa123")  # TODO modify help
    parser.add_argument("--out_folder_name", type=str, default="uie_preprocess", help="aaa123")  # TODO modify help
    args = parser.parse_args()

    # setting
    uie = Taskflow("information_extraction", task_path=args.uie_model_name_or_path, schema=SCHEMA, precision="fp16")
    args.dynamic_adjust_length = eval(args.dynamic_adjust_length)
    tokenizer = AutoTokenizer.from_pretrained(args.utc_model_name_or_path)
    special_word_len = get_template_tokens_len(tokenizer, os.path.join(args.dataset_path, "label.txt"))
    max_content_len = args.max_seq_len - special_word_len
    output_path = os.path.join(args.dataset_path, args.out_folder_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # processing
    fail_examples = {}
    fail_examples["total_counter"] = 0
    total = 0
    for data_name in (args.train_file, args.dev_file, args.test_file):
        out_text = []
        fail_examples[data_name] = {}
        fail_examples[data_name]["fail_counter"] = 0
        fail_examples[data_name]["fail_examples"] = []
        logger.info(f"Start preprocessing {data_name}...")
        number_of_examples = 0
        with open(os.path.join(args.dataset_path, data_name), "r", encoding="utf8") as fp:
            number_of_examples = len(fp.readlines())
        total += number_of_examples

        with open(os.path.join(args.dataset_path, data_name), "r", encoding="utf8") as fp:
            for example in tqdm(fp, total=number_of_examples):
                example = json.loads(example.strip())
                uie_output = uie(example["text_a"])
                if DOUBLE_CHECK:
                    do_double_check(uie_output, example["labels"])
                new_text = filter_text(
                    raw_text=example["text_a"],
                    uie_output=uie_output,
                    max_len_of_new_text=max_content_len,
                    threshold=args.threshold,
                    dynamic_adjust_length=args.dynamic_adjust_length,
                )

                if len(new_text) == 0 and len(example["labels"]) > 0:
                    fail_examples[data_name]["fail_counter"] += 1
                    fail_examples["total_counter"] += 1
                    fail_examples[data_name]["fail_examples"].append(example)
                    logger.debug(fail_examples["total_counter"])
                    continue
                elif len(new_text) == 0 and len(example["labels"]) == 0:
                    new_text = example["text_a"][:10]

                example["text_a"] = new_text
                out_text.append(example)

        write_json(out_text, out_path=os.path.join(output_path, data_name))
        logger.info(f"Finish {data_name} processing. Total samples: {len(out_text)}.")
        logger.info(f"Fail in {data_name} samples: {fail_examples[data_name]['fail_counter']}.")

    logger.debug(f"Fail samples: {fail_examples}")
    logger.info(f"Total fail {fail_examples['total_counter']} examples in {total}.")
    logger.info(f"UIE Missing Rate: {fail_examples['total_counter']/total}.")
    logger.info(f"Finish all preprocessing.")

    logger.debug(f"沒抓到 {missing_recode_delete_me}")
    logger.debug(f"抓錯 {error_recode_delete_me}")
    logger.debug(f"Total: {total_delete_me}")
    for each_schema in SCHEMA:
        logger.debug(f"{each_schema} 的沒抓到率: {missing_recode_delete_me[each_schema]/total_delete_me[each_schema]}.")

    # shutil.copyfile(os.path.join(args.dataset_path, "label.txt"), output_path + "label.txt")
