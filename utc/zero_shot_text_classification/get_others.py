import json
import pandas as pd

if __name__ == "__main__":

    label_studio_file = "labelstudio_data/formal_data/classification.json"
    with open(label_studio_file, "r", encoding="utf-8") as f:
        raw_examples = json.loads(f.read())

    inner_ids = []
    samples = []
    choices = []
    for example in raw_examples:
        if example["annotations"][0]["result"]:
            if "其他" in example["annotations"][0]["result"][0]["value"]["choices"]:
                inner_ids.append(example["inner_id"])
                samples.append(example["data"]["jid"])
                choices.append(example["annotations"][0]["result"][0]["value"]["choices"])


    df = pd.DataFrame({
        "inner_id":inner_ids,
        "jid":samples,
        "choices":choices
    }
    )
            

    # breakpoint()

    df.to_csv("labelstudio_data/anomaly_data.csv")
    
    