import os
import json
from collections import defaultdict

# test
def collect_label_information(
    label_file: str = "/home/ubuntu/work_debug/verdict-cls-debug/utc/zero_shot_text_classification/data/label.txt",
):
    label_name = []
    label_count = {}
    with open(label_file, "r") as f:
        for i, name in enumerate(f):
            label_count[i] = 0
            label_name.append(name.strip())
    return label_name, label_count


currentdir = os.getcwd()
filedir = "verdict-cls-debug/utc/zero_shot_text_classification/data"
data_name = ["train", "dev", "test"]
total_num_of_label = 55
all_data = defaultdict(list)
label_name, tmp = collect_label_information()
label_distribution = {i: tmp.copy() for i in data_name + ["total"]}


for each_data in data_name:
    dir = os.path.join(currentdir, filedir, each_data + ".txt")
    with open(dir, "r", encoding="utf8") as f:
        # tmp = f.readlines()
        for i in f:
            this_json = json.loads(i)
            all_data[each_data].append(this_json)
            for each_label in this_json["labels"]:
                label_distribution[each_data][each_label] += 1
                label_distribution["total"][each_label] += 1


import numpy as np
import matplotlib.pyplot as plt

label_distribution["dev"]

# creating the dataset
data = label_distribution["train"]
courses = list(data.keys())
values = list(data.values())

fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(courses, values, color="maroon", width=0.4)

plt.xlabel("Courses offered")
plt.ylabel("No. of students enrolled")
plt.title("Students enrolled in different courses")
plt.show()
