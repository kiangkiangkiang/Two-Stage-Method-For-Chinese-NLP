import os
import json
import pandas as pd
from paddlenlp.utils.log import logger
from typing import List


def read_all_json_data(data_folder: str = "./labelstudio_data") -> List[dict]:
    all_json = [file for file in os.listdir(data_folder) if file[-5:] == ".json"]
    all_data = []
    for json_ in all_json:
        with open(data_folder + "/" + json_, "r", encoding="utf8") as f:
            logger.info(f"Read {json_}...")
            f_r = f.readlines()
            for i in f_r:
                f_json = json.loads(i)
                all_data.extend(f_json)
    return all_data


def clear_data_duplicate(data: List[dict]):
    all_jid = []
    new_data = []
    num_of_duplicate = 0
    for each_data in data:
        if each_data["data"]["jid"] not in all_jid:
            new_data.append(each_data)
            all_jid.append(each_data["data"]["jid"])
        else:
            num_of_duplicate += 1
            logger.warning(f"Duplicate Case:")
            logger.warning(f"{each_data['data']}")
            logger.warning(f"Total Duplicate: {num_of_duplicate}")

    logger.info(f"Clear All Duplicate Cases: {num_of_duplicate}")
    logger.info(f"Length of Remain Data: {len(new_data)}")
    return new_data


def write_data(data, out_dir: str = "./labelstudio_data", file_name: str = "label_studio_output.json"):
    with open(out_dir + "/" + file_name, "w", encoding="utf-8") as outfile:
        jsonString = json.dumps(data, ensure_ascii=False)
        outfile.write(jsonString)


if __name__ == "__main__":
    logger.info("Start Read and Merge All Data...")
    all_json_data = read_all_json_data()
    logger.info(f"End Read Data Session...")
    logger.info(f"Length of All Data: {len(all_json_data)}")

    logger.info("Start Check Data Duplicate...")
    all_json_data = clear_data_duplicate(all_json_data)
    logger.info("End Clear Data Duplicate...")

    logger.info("Start Write Data...")
    write_data(all_json_data)
    logger.info("End Write Data...")

    logger.info("Finish Data Preprocessing.")
