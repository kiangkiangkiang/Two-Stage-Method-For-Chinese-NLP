import os
import json
import numpy as np

dir = os.path.join(os.getcwd(), "verdict-cls-debug/utc/zero_shot_text_classification/labelstudio_data/")

out_num = 30
options = []
label_list = []
outfile = []

with open(os.path.join(dir, "label.txt"), "r", encoding="utf8") as f:
    options = [i.strip() for i in f]

with open(os.path.join(dir, "label_studio_output.json"), "r", encoding="utf8") as f:
    for file in f:
        for example in json.loads(file.strip()):
            # example['annotations'][0]['result']
            if example["annotations"][0]["result"]:
                for raw_label in example["annotations"][0]["result"][0]["value"]["choices"]:
                    if raw_label not in options:
                        breakpoint()
                        raise ValueError(
                            f"Label `{raw_label}` not found in label candidates `options`. Please recheck the data."
                        )
                    label_list.append(np.where(np.array(options) == raw_label)[0].tolist()[0])

            do_out = [False, False, False]
            for label in label_list:
                if label < 36:
                    do_out[0] = True
                elif label >= 36 and label < 47:
                    do_out[1] = True
                else:
                    do_out[2] = True

            if np.all(do_out):
                outfile.append(example)

            if len(outfile) == out_num:
                break

with open(
    os.path.join(os.getcwd(), "verdict-cls-debug/utc/wait_for_label_for_uie/file.json"), "w", encoding="utf-8"
) as f:
    jsonString = json.dumps(outfile, ensure_ascii=False)
    f.write(jsonString)
