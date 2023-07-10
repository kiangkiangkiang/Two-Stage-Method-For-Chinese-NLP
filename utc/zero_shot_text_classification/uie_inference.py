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
from paddlenlp.transformers import AutoTokenizer
import re

SCHEMA = ["原告年齡", "肇事過失責任比例", "受有傷害"]
IS_DOUBLE_CHECK = True
IS_RULE_BASED_POSTPROCESSING = True

# TODO delete do_double_check (this for debug)
missing_recode_delete_me = {entity_type: 0 for entity_type in SCHEMA}
error_recode_delete_me = missing_recode_delete_me.copy()
total_delete_me = missing_recode_delete_me.copy()
default_counter = missing_recode_delete_me.copy()
fail_examples = {}
fail_examples["total_counter"] = 0


def double_check(uie_output):
    label_counter = {entity_type: 0 for entity_type in SCHEMA}
    found_label = label_counter.copy()

    for key in uie_output[0]:
        found_label[key] += 1

    for each_schema in SCHEMA:
        if label_counter[each_schema] > 0 and found_label[each_schema] == 0:
            missing_recode_delete_me[each_schema] += 1
            logger.debug(f"UIE 沒抓出來 「{each_schema}. Total: {missing_recode_delete_me[each_schema]}")
        elif label_counter[each_schema] == 0 and found_label[each_schema] > 0:
            error_recode_delete_me[each_schema] += 1
            logger.debug(f"UIE 抓錯 「{each_schema}. Total: {error_recode_delete_me[each_schema]}")


"""
{'受有傷害': [{'text': '右側', 'start': 452, 'end': 454, 'probability': 0.565}, 
             {'text': '左側手肘擦傷', 'start': 445, 'end': 451, 'probability': 0.83764506836}, 
             {'text': '右側', 'start': 442, 'end': 444, 'probability': 0.7103648}, 
             {'text': '右側', 'start': 462, 'end': 464, 'probability': 0.5141215855},
             {'text': '左側膝部擦傷', 'start': 465, 'end': 471, 'probability': 0.74013138}, 
             {'text': '左側髖部挫傷', 'start': 455, 'end': 461, 'probability': 0.853}]
} 
"""


class rule_based_processer:
    def __init__(self) -> None:
        self.default_uie_format = [
            {
                "text": "預設輸出",
                "start": 0,
                "end": 1,
                "probability": 0.5,  # any float is ok
            }
        ]

    def __re_to_uie_format(self, re_result):
        simulate_uie_output = []
        for result in re_result:
            simulate_uie_output.append(
                {
                    "text": result.group(),
                    "start": result.start(),
                    "end": result.end(),
                    "probability": 0.5,  # any float is ok
                }
            )
        return simulate_uie_output

    def add_age_information_index(self, raw_text):
        result = []
        # stage 1
        pattern = ".[^65]歲|出生|年.{0,5}[^發|^產]生|年次|年紀"
        for match in re.finditer(pattern, raw_text):
            result.append(match)
        if result:
            return self.__re_to_uie_format(result)

        # stage 2
        pattern = "歲|退休|學生|小學|國中|高中|大學|研究所|畢業|未成年|原告.{0,15}年}"
        for match in re.finditer(pattern, raw_text):
            result.append(match)

        return self.__re_to_uie_format(result) if result else self.default_uie_format

    def add_responsibility_information_index(self, raw_text):
        # stage 1
        result = []
        pattern0 = ["%.{0,20}過失|過失.{0,20}%", "%.{0,20}責任|責任.{0,20}%", "%.{0,20}肇事|肇事.{0,20}%"]
        pattern1 = [i.replace("%", "分之") for i in pattern0]
        pattern2 = [i.replace("%", "％") for i in pattern0]
        pattern3 = [i.replace("%", "/") for i in pattern0]
        pattern = "".join([u + "|" for i in (pattern0, pattern1, pattern2, pattern3) for u in i])[:-1]
        for match in re.finditer(pattern, raw_text):
            result.append(match)
        if result:
            return self.__re_to_uie_format(result)

        # stage 2
        pattern = "比例|%|肇事|責任|過失|％|[^部]分之"
        for match in re.finditer(pattern, raw_text):
            result.append(match)

        return self.__re_to_uie_format(result) if result else self.default_uie_format

    def add_injury_information_index(self, raw_text):
        # stage 1
        pattern = "受有.{0,100}傷害|受有.{0,100}傷勢"
        result = re.search(pattern=pattern, string=raw_text)
        if result:
            return self.__re_to_uie_format([result])

        # stage 2
        pattern = "受有|傷"
        result = []
        for match in re.finditer(pattern, raw_text):
            result.append(match)

        return self.__re_to_uie_format(result) if result else self.default_uie_format

    # 當 label 長度或順序改變 這個 fun 會有問題
    def postprocessing(self, raw_text, uie_output):
        postprocessing_function_set = (
            self.add_age_information_index,
            self.add_responsibility_information_index,
            self.add_injury_information_index,
        )
        postprocessing = {label_type: do_fun for label_type, do_fun in zip(SCHEMA, postprocessing_function_set)}
        has_label_type = {label_type: False for label_type in SCHEMA}

        for label_type in SCHEMA:
            if uie_output[0].get(label_type) is None and has_label_type[label_type]:
                uie_output[0][label_type] = postprocessing[label_type](raw_text=raw_text)
                if uie_output[0][label_type][0]["text"] == "預設輸出":
                    default_counter[label_type] += 1
                    logger.debug(f"{label_type} default: default_counter[label_type]")

        return uie_output


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
    processer = rule_based_processer()

    parser.add_argument(
        "--dataset_path",
        default="./data",
        type=str,
        help="Local dataset directory including train.txt, dev.txt and label.txt (optional).",
    )

    parser.add_argument("--data_file_to_inference", type=str, default="data.json", help="Train dataset file name.")
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
    uie = Taskflow("information_extraction", task_path=args.uie_model_name_or_path, schema=SCHEMA)
    args.dynamic_adjust_length = eval(args.dynamic_adjust_length)
    tokenizer = AutoTokenizer.from_pretrained(args.utc_model_name_or_path)
    special_word_len = get_template_tokens_len(tokenizer, os.path.join(args.dataset_path, "label.txt"))
    max_content_len = args.max_seq_len - special_word_len
    output_path = os.path.join(args.dataset_path, args.out_folder_name)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # processing
    total = 0
    out_text = []
    fail_examples["inference_data"] = {}
    fail_examples["inference_data"]["fail_counter"] = 0
    fail_examples["inference_data"]["fail_examples"] = []
    logger.info(f"Start preprocessing inference_data...")
    number_of_examples = 0
    with open(args.data_file_to_inference, "r", encoding="utf8") as fp:
        data = json.load(fp) 
        number_of_examples = len(data)

    for example in tqdm(data, total=number_of_examples):

        uie_output = uie(example["jfull_compress"])

        if IS_DOUBLE_CHECK:
            double_check(uie_output)

        if IS_RULE_BASED_POSTPROCESSING:
            uie_output = processer.postprocessing(
                raw_text=example["jfull_compress"], uie_output=uie_output
            )

        new_text = filter_text(
            raw_text=example["jfull_compress"],
            uie_output=uie_output,
            max_len_of_new_text=max_content_len,
            threshold=args.threshold,
            dynamic_adjust_length=args.dynamic_adjust_length,
        )

        if len(new_text) == 0:
            new_text = example["jfull_compress"][:10]

        example["jfull_compress"] = new_text
        out_text.append(example)

    write_json(out_text, out_path=os.path.join(output_path, 'processed_data.json'))
    logger.info(f"Finish inference_data preprocessing. Total samples: {len(out_text)}.")

    logger.info(f"Finish all preprocessing.")

    if IS_DOUBLE_CHECK:
        logger.debug(f"沒抓到 {missing_recode_delete_me}")
        logger.debug(f"抓錯 {error_recode_delete_me}")
        logger.debug(f"Total: {total_delete_me}")

    logger.debug(f"規則也抓不到而走到 default 的筆數: {default_counter}")

    for each_schema in SCHEMA:
        logger.debug(f"{each_schema} 的沒抓到率: {missing_recode_delete_me[each_schema]/total_delete_me[each_schema]}.")

    # shutil.copyfile(os.path.join(args.dataset_path, "label.txt"), output_path + "label.txt")
