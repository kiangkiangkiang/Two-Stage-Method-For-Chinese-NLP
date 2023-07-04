import pandas as pd
import numpy as np
import json
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score

NUM_LABELS = 55
NUM_TEST = 100

index_table = {
        "頭頸部": 0,
        "臉": 1,
        "胸部": 2,
        "腹部": 3,
        "背部": 4,
        "骨盆": 5,
        "上肢": 6,
        "下肢": 7,
        "其他體傷部位": 8,
        "骨折": 9,
        "骨裂": 10,
        "擦挫傷": 11,
        "撕裂傷": 12,
        "鈍傷": 13,
        "損傷": 14,
        "胸部損傷": 15,
        "神經損傷": 16,
        "拉傷": 17,
        "扭傷": 18,
        "灼傷": 19,
        "脫位": 20,
        "壓迫": 21,
        "破缺損": 22,
        "腦震盪": 23,
        "壞死": 24,
        "內出血": 25,
        "水腫": 26,
        "瘀血": 27,
        "栓塞": 28,
        "剝離": 29,
        "截肢": 30,
        "衰竭": 31,
        "休克": 32,
        "失能": 33,
        "死亡": 34,
        "其他體傷型態": 35,
        "肇責 0/100": 36,
        "肇責 10/90": 37,
        "肇責 20/80": 38,
        "肇責 30/70": 39,
        "肇責 40/60": 40,
        "肇責 50/50": 41,
        "肇責 60/40": 42,
        "肇責 70/30": 43,
        "肇責 80/20": 44,
        "肇責 90/10": 45,
        "肇責 100/0": 46,
        "未滿18歲(高中以下)": 47,
        "18-24歲(大學、研究所)": 48,
        "25-29歲": 49,
        "30-39歲": 50,
        "40-49歲": 51,
        "50-59歲": 52,
        "60-64歲": 53,
        "65歲以上(退休)": 54
    }

def human_label_to_index(human_labels):

    multilabel_true = [0] * NUM_LABELS

    for index in human_labels:
        multilabel_true[index] = 1

    return multilabel_true


def gpt_label_to_index(gpt_labels):

    gpt_labels = eval(gpt_labels)
    labels_pred_as_true = []
    
    for key in gpt_labels.keys():
        sublabels = gpt_labels[key]

        for k, v in sublabels.items(): 
            if v:
                if key == "體傷部位" and k == "其他":
                    labels_pred_as_true.append("其他體傷部位")

                elif key == "體傷型態" and k == "其他":
                    labels_pred_as_true.append("其他體傷型態")

                else:
                    labels_pred_as_true.append(k)

    multilabel_pred = [0] * NUM_LABELS

    for label in labels_pred_as_true:
        index_pred_as_true = index_table[label]
        multilabel_pred[index_pred_as_true] = 1

    return multilabel_pred


if __name__ == '__main__':

    testset_path = "data/testset.txt"
    gpt_label_path = "metadata/summary_gpt-3.5-turbo-0301_format_gpt-4_cls.csv"

    with open(testset_path, "r", encoding="utf-8") as textfile:
        testsets = textfile.readlines()

    gpt_data = pd.read_csv(gpt_label_path)

    y_true = []
    y_pred = []

    for i in range(NUM_TEST):
        testset = eval(testsets[i].strip())
        human_labels = testset["labels"]
        multilabel_true = human_label_to_index(human_labels)
        y_true.append(multilabel_true)

        multilabel_pred = gpt_label_to_index(gpt_data['cls_label'][i])
        y_pred.append(multilabel_pred)

    
    print("acc: ", accuracy_score(y_true, y_pred))
    print("precision: ", precision_score(y_true, y_pred, average="micro"))
    print("recall: ",recall_score(y_true, y_pred, average="micro"))
    print("micro f1: ",f1_score(y_true, y_pred, average="micro"))
    print("macro f1: ",f1_score(y_true, y_pred, average="macro"))


    print("體傷部位: ", f1_score([y[:9] for y in y_true], [y[:9] for y in y_pred], average="micro"))
    print("體傷型態 ", f1_score([y[9:36] for y in y_true], [y[9:36] for y in y_pred], average="micro"))
    print("肇責: ",f1_score([y[36:47] for y in y_true], [y[36:47] for y in y_pred], average="micro"))
    print("年齡: ",f1_score([y[47:] for y in y_true], [y[47:] for y in y_pred], average="micro"))
