# Chinese Verdict NLU

<br>

## Source of data
[台灣司法院查詢系統](https://judgment.judicial.gov.tw/FJUD/default.aspx)


## Model API
1. gpt-3.5-turbo-0301
2. gpt-4

## API

1. cls task
```
python gpt4nlu.py --list_nlp_task 'cls'
```

1. uie task
```
python gpt4nlu.py --list_nlp_task 'uie'
```