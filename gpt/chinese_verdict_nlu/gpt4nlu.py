import os
import re
import openai
import argparse
import pandas as pd
from tqdm import tqdm
import time


def unwrap_keys(obj, prefix='', keys=None):
    if keys is None:
        keys = []
        
    for key, value in obj.items():
        current_key = f'{prefix}{key}' if prefix else key
        keys.append(current_key)
        
        if isinstance(value, dict):
            unwrap_keys(value, prefix=f'{current_key}.', keys=keys)
    return keys

def messeage_prepare(text, prompt, system_info):
    mess = prompt + text
    message = [
        {"role": "system", "content": system_info},
        {"role": "user", "content": mess}
        ]
    return message

def openai_chat_inference_and_calculate(
        model, 
        text,
        prompt_stage, 
        system_info,
        tokens_info,
        temperature=0.0
        ):

    # try:
    response = openai.ChatCompletion.create(
        model=model,
        messages=messeage_prepare(text, prompt=prompt_stage, system_info=system_info),
        temperature=temperature,
        )

    # except:
    #     print("The server is overloaded")
    #     breakpoint()

    completions = response["choices"][0]["message"]["content"]
    tokens_info["prompt_tokens"] += response["usage"]["prompt_tokens"]
    tokens_info["completion_tokens"] += response["usage"]["completion_tokens"]
    tokens_info["total_tokens"] += response["usage"]["total_tokens"]
        
    return completions, tokens_info

def cls_list_of_str_category_to_dict(list_of_str_category, cls_key, generate_prompt=True):
    str_of_bool = ':True或False' if generate_prompt else ':False'
        
    str_sub_category_list = ['"' + sub_key + '"' + str_of_bool for sub_key in list_of_str_category]
    str_sub_category = ','.join(str_sub_category_list)
    str_category = '"' + cls_key + '":' + '{' + str_sub_category + '},'
        
    return str_category

def uie_list_of_str_category_to_dict(list_of_str_category, uie_key, generate_prompt=True):
    str_of_bool = ':True或False' if generate_prompt else ':False'
 
    str_sub_category_list = ['"' + sub_key + '"' + ':' + '"擷取原文' + sub_key + '字串or空字串"' for sub_key in list_of_str_category]
    str_sub_category_list.append('"未提及"'+ str_of_bool)
    str_sub_category = ','.join(str_sub_category_list)
    str_category = '"' + uie_key + '"' + ':' + '{' + str_sub_category + '},'
        
    return str_category

def prompt_and_default_label_prepare_for_cls_and_uie(cls_category_dict=None, uie_category_dict=None):
    str_category_for_all_task = "[{"
    str_category_for_default_label = "{"

    if cls_category_dict:
        for cls_key in cls_category_dict:
            list_of_str_category = cls_category_dict[cls_key]
            str_category_for_all_task += cls_list_of_str_category_to_dict(list_of_str_category, cls_key, generate_prompt=True)
            str_category_for_default_label += cls_list_of_str_category_to_dict(list_of_str_category, cls_key, generate_prompt=False)

    if uie_category_dict:
        for uie_key in uie_category_dict:
            list_of_str_category = uie_category_dict[uie_key]
            str_category_for_all_task += uie_list_of_str_category_to_dict(list_of_str_category, uie_key, generate_prompt=True)
            str_category_for_default_label += uie_list_of_str_category_to_dict(list_of_str_category, uie_key, generate_prompt=False)
        
    str_category_for_all_task += "}]"
    str_category_for_default_label += "}"

    prompt_label = str_category_for_all_task 
    default_label = eval(str_category_for_default_label)
    
    return prompt_label, default_label

def prepare_prompt(additional_and_desc_dict, target_and_desc_dict, prompt_label, prompt_type=None):
    target_and_desc = ",\n".join([key + "(" + value + "):未提及" if value != None else key for key,value in target_and_desc_dict.items()])
    additional_and_desc = ",\n".join([key + "(" + value + "):未提及" if value != None else key for key,value in additional_and_desc_dict.items()])
    
    if prompt_type == "summary":
        system_info = "幫使用者從[文本]中擷取指定資訊,回答越精要越好"
        prompt_stage = "更新[已知訊息]中的項目,最後以下面的項目條列回答:\n" + additional_and_desc + "\n" + target_and_desc + "\n"
    
    elif prompt_type == "clean":
        system_info = "去除重複資訊,去除重複回答,回答越精要越好"
        prompt_stage = "指定項目在[目標格式]中,從原文擷取需要資訊:\n[目標格式]\n" + target_and_desc + "\n"
    
    elif prompt_type == "format":
        system_info = "根據提供格式產生結構化資料,回答越符合格式越好"
        prompt_stage = "將原文資訊依照格式要求填入下方格式:\n" + prompt_label + "\n"
        
    return system_info, prompt_stage
    

def get_text_index(text, target):
    index = [(i.start(0), i.end(0)) for i in re.finditer(target, text)] if target != '' else []
    return index


def main(args):     
    f = open("openai_api.txt", "r")
    api_key = f.readline().replace("\n","")
    openai.api_key = api_key

    #### Config
    target_and_desc_dict={}
    additional_and_desc_dict={}
    if "cls" in args.list_nlp_task:
        target_and_desc_dict.update({
            "體傷部位":"受傷部位,任何人",
            "體傷型態":"受傷型態,任何人",
            "原告是否大學生":"是or否",
            "原告是否已退休":"是or否",
            "原告是否未成年":"是or否",
            "原告年歲,年齡":"判決年份-原告出生年份",
            "原告肇責比例":"被告負全責則為0%",
        })

        additional_and_desc_dict.update({
            "判決年份":None,
            "原告出生年份":"可能為無",
        })
        cls_category_dict = {
            "體傷部位":['頭頸部', '臉', '胸部', '腹部', '背部', '骨盆', '上肢', '下肢', '其他體傷部位'],
            "體傷型態":['骨折', '骨裂', '擦挫傷', '撕裂傷', '鈍傷', '損傷', '胸部損傷', '神經損傷', '拉傷', '扭傷', '灼傷', '脫位', '壓迫', '破缺損', '腦震盪', '壞死', '內出血', '水腫', '瘀血', '栓塞', '剝離', '截肢', '衰竭', '休克', '失能', '死亡', '其他體傷型態'],
            "原告肇責比例":['肇責 0/100', '肇責 10/90', '肇責 20/80', '肇責 30/70', '肇責 40/60', '肇責 50/50', '肇責 60/40', '肇責 70/30', '肇責 80/20', '肇責 90/10', '肇責 100/0'],
            "原告年齡":['未滿18歲(高中以下)', '18-24歲(大學、研究所)', '25-29歲', '30-39歲', '40-49歲', '50-59歲', '60-64歲', '65歲以上(退休)'],
            }
    else:
        cls_category_dict = None
        
    if "uie" in args.list_nlp_task:
        target_and_desc_dict.update({
            "法院認為適當的精神慰撫金額":"非財產損害賠償,一個金額,擷取原文,0元=無",
            "法院認為適當的醫療費用":"一個金額,擷取原文",
            "原告月薪":"去除前綴詞,一個金額,擷取原文",
        })
        additional_and_desc_dict.update({
            "原告or上訴人提出的精神慰撫金額":"非財產損害賠償,一個金額,擷取原文",
            "原告or上訴人提出的醫療費用":"一個金額,擷取原文",
            "原告日薪":"去除前綴詞,一個金額,擷取原文",
            "原告年薪":"去除前綴詞,一個金額,擷取原文",
        })
        uie_category_dict = {
            "精神慰撫金額":["金額"],
            "醫療費用":["金額"],
            "原告月薪":["金額"]
            }
    else:
        uie_category_dict = None
        
    word_eliminated_in_label = ["\n","新台幣","新臺幣"]

    prompt_label, default_label = prompt_and_default_label_prepare_for_cls_and_uie(cls_category_dict, uie_category_dict)
    system_info_summary, prompt_stage_summary = prepare_prompt(additional_and_desc_dict, target_and_desc_dict, prompt_label, prompt_type="summary")
    system_info_clean, prompt_stage_clean = prepare_prompt(additional_and_desc_dict, target_and_desc_dict, prompt_label, prompt_type="clean")
    system_info_format, prompt_stage_format = prepare_prompt(additional_and_desc_dict, target_and_desc_dict, prompt_label, prompt_type="format")
    word_eliminated_in_label_pattern = "|".join(word_eliminated_in_label)

    #### Initial status, if past info exist, read and keep process
    start_idx = 0
    dataframe_info ={
        "verdict":[],
        "summary":[],
    }
    if cls_category_dict:
        dataframe_info["cls_label"] = []
    if uie_category_dict:
        dataframe_info["uie_label"] = []
        
    cost_dict = {
        "gpt-3.5-turbo-0301":0.06,
        "gpt-3.5-turbo-0613":0.06,
        "gpt-4":1.8,
    }
    chunk_token_length_dict = {
        "gpt-3.5-turbo-0301":1800,
        "gpt-3.5-turbo-0613":1800,
        "gpt-4":20000,
    }
    tokens_info_summary = {
        "prompt_tokens":0,
        "completion_tokens":0,
        "total_tokens":0,
    }
    tokens_info_format = {
        "prompt_tokens":0,
        "completion_tokens":0,
        "total_tokens":0,
    }

    summary_chunk_token_overlap_token = args.chunk_overlap_token
    
    require_keys = unwrap_keys(default_label)
    saved_file_path = os.path.join(args.path_to_file,"summary_"+args.summary_model+"_format_"+args.format_model+"_"+"_".join(args.list_nlp_task)+".csv")
    if os.path.exists(saved_file_path):
        df = pd.read_csv(saved_file_path)
        start_idx = len(df)
        dataframe_info = df.to_dict('list')
        
    
    with open(args.dataset, "r") as json_file:
        list_of_str_json = json_file.readlines()
        list_of_dict = [ eval(js) for js in list_of_str_json ]
    
    list_id_verdict_label = [ key for key in args.list_id_verdict_label]
    verdict_input_dict = {key:[d[key] for d in list_of_dict] for key in list_id_verdict_label}
     
    epoch_bar = tqdm(range(start_idx, args.batch_size), desc="Calling OpenAI API...")

    for idx in range(start_idx, args.batch_size):
        
        #### Choose model
        verdict = verdict_input_dict[list_id_verdict_label[1]][idx]
        human_label = verdict_input_dict[list_id_verdict_label[2]][idx]
        if args.check_inner_output:
                print(f"Human Label : {human_label}")
        
        summary_model = args.summary_model
        format_model  = args.format_model
        
        summary_chunk_token_length = chunk_token_length_dict[summary_model]
        summary_cost_per_1k_token = cost_dict[summary_model]
        summary_chunk_token_start_idx = summary_chunk_token_length - summary_chunk_token_overlap_token
        
        #### Stage 1: Summarizing chunk
        idx_split_verdict_start_iterator = range(0, len(verdict), summary_chunk_token_start_idx)
        epoch_bar_chunk = tqdm(idx_split_verdict_start_iterator, desc=f"Summary chunk with {summary_model}")

        completions =  ""
        for idx_split_verdict_start in idx_split_verdict_start_iterator:
            if (idx_split_verdict_start+summary_chunk_token_length+1) > len(verdict):
                idx_split_verdict_start = len(verdict) - summary_chunk_token_length
            idx_split_verdict_end = idx_split_verdict_start+summary_chunk_token_length
            split_verdict = verdict[idx_split_verdict_start:idx_split_verdict_end]
            
            completions_with_text = "\n[已知訊息]\n" + completions + "\n[文本]:" + split_verdict

            completions, tokens_info = openai_chat_inference_and_calculate(
                summary_model, 
                completions_with_text, 
                prompt_stage_summary,
                system_info_summary,
                tokens_info_summary,
                args.base_temp
                )
            
            epoch_bar_chunk.set_postfix({
                      "p_token": tokens_info_summary["prompt_tokens"],
                      "c_token": tokens_info_summary["completion_tokens"],
                      "total_token": tokens_info_summary["total_tokens"],
                      f"{summary_model} Cost NTD":tokens_info_summary["total_tokens"]//1000*summary_cost_per_1k_token,
            })
            epoch_bar_chunk.update()
            
            if args.check_inner_output:
                print(f"Chunk summary : {completions}")

        epoch_bar_chunk.close()
            
        #### Stage 2:Generate specific format
        format_cost_per_1k_token = cost_dict[format_model]
  
        epoch_bar_clean = tqdm(range(1), desc=f"Clean summary with {format_model}")
        completions_stage_clean, tokens_info = openai_chat_inference_and_calculate(
                format_model, 
                completions, 
                prompt_stage_clean,
                system_info_clean,
                tokens_info,
                args.base_temp
            )
            
        epoch_bar_clean.update()

        if args.check_inner_output:
            print(f"Chunk summary : {completions_stage_clean}")

        epoch_bar_clean.close()

        retry_step = 0
        schema_check_fail = True
        epoch_bar_retry = tqdm(range(args.retry_size), desc=f"Generating format with {format_model}") 

        while retry_step <= args.retry_size and schema_check_fail:
            label, tokens_info = openai_chat_inference_and_calculate(
                format_model, 
                completions_stage_clean, 
                prompt_stage_format,
                system_info_format,
                tokens_info_format,
                args.base_temp + 0.1*retry_step
                )
            
            try:
                text = re.sub(word_eliminated_in_label_pattern, "", label)
                st, ed = re.search("\[", text).end(), re.search("\]", text).start()
                clean_label = eval(text[st:ed])
                schema_check_fail = require_keys != unwrap_keys(clean_label)
                
            except:
                pass
            
            epoch_bar_retry.update()
            retry_step += 1
     
        if schema_check_fail == True:
            clean_label = default_label
           
        epoch_bar_retry.close()

        print(f"Final Label {clean_label}")

        if cls_category_dict:
            cls_label = {cls_key:clean_label[cls_key] for cls_key in cls_category_dict}
            dataframe_info["cls_label"].append(cls_label)
            
        if uie_category_dict:
            uie_label = []
            for uie_key,uie_category in uie_category_dict.items():
                for category in uie_category:
                    if not clean_label[uie_key]["未提及"]:
                        target_index = get_text_index(verdict, clean_label[uie_key][category])
                        for start, end in target_index:
                            uie_label.append({"start":start,"end":end,"text":clean_label[uie_key][category],"labels":[uie_key]})
            dataframe_info["uie_label"].append(uie_label)
            
        dataframe_info["verdict"].append(verdict)
        dataframe_info["summary"].append(completions_stage_clean)
        epoch_bar.update()
        epoch_bar.set_postfix({
                        "p_token": tokens_info_format["prompt_tokens"],
                        "c_token": tokens_info_format["completion_tokens"],
                        "total_token": tokens_info_format["total_tokens"],
                        f"This article total cost NTD":tokens_info_summary["total_tokens"]//1000*summary_cost_per_1k_token
                                                   +tokens_info_format["total_tokens"]//1000*format_cost_per_1k_token,
        })
        
        #### Save file after n prompt
        if (idx+1) % args.save_step == 0:
            df = pd.DataFrame(dataframe_info)
            df.to_csv(saved_file_path, header=True, index=False)

    epoch_bar.close()

    
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='data/formatted_testset.jsonl')
    parser.add_argument('--summary_model', type=str, default='gpt-3.5-turbo-0301')
    parser.add_argument('--format_model', type=str, default='gpt-4')
    parser.add_argument('--list_nlp_task', type=str, nargs='*', default=['cls', 'uie'])
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--retry_size', type=int, default=3)
    parser.add_argument('--list_id_verdict_label', type=str, nargs='*', default=['id', 'data', 'label'])
    parser.add_argument('--check_inner_output', type=bool, default=True)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--chunk_overlap_token', type=float, default=25)
    parser.add_argument('--base_temp', type=float, default=0.0)
    parser.add_argument('--path_to_file', type=str, default='metadata')
    args = parser.parse_args()
    main(args)

