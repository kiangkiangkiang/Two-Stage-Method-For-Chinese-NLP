# Two-Stage Chinese Verdict Classification (二階段臺灣司法判決書分析)

--- 


## Guide


| **Section**                                      |  **Description** | 
|:------------------------------------------------:|:----------------:|
|<a href=#Introduction> Introduction </a>          | Introduction to the method and long sequence issue| 
|<a href=#Two-Stage Architecture> Architecture </a>| Detail the architecture of two-stage method and scenario|
|<a href=#Results> Results </a>                    | Methods comparison on long sequence       |
|<a href=#Quick Start> Quick Start </a>            | Implement the two-stage analysis|




## Introduction

In this work, we present a **two-stage Chinese verdict classification method** to overcome the **long sequence issue ** on Chinese verdict tasks. The two-stage method reaches better performance than the recurrent model like XLNet. 


In the past, addressing the issue of long sequence might involve using recurrent models or truncate the sequence. However, we have discovered that our proposed two-stage model can effectively **extract important information**, which can then be **fed into any downstream tasks**. This approach proves to be a viable solution for overcoming the challenges posed by long texts.

## Two-Stage Architecture

Our designed two-stage method is illustrated in the following diagram. This method can overcome challenges posed by any long sequence and only requires replacing the downstream task model.


![](https://hackmd.io/_uploads/Syy1RM_wn.png)


The key to this architecture lies in the [UIE (Universal Information Extraction)](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie) model, which is presented by [PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP/tree/develop) and has shown excellent performance on Chinese NLP tasks such as NER (Named Entity Recognition) or RE (Relation Extraction). We use UIE model to extract the important information and collect the information to feed the downstream task model.

### Stage 1: Extract Information

⚡ **Information Extraction**: We first use UIE model to extract the important information in each content and get the index of each information.

⚡ **Index Windows**: Secondly, locate the index of important information, and open the windows on each index to extract the important paragraph.

⚡ **Aggregation**: Finally, aggregate all important paragraph for downstream task.


### Stage 2: Downstream task

After obtaining the paragraph containing important information, we get a **shorter sequence**, effectively addressing the issue of long sequence that models struggle to handle. 

The downstream task is then determined based on the user's area of interest or focus. 

In the next chapter, we will discuss how this method can be applied to address the issue of analyzing Chinese verdicts.

### Experiment on Two-Stage Method

Through this architecture, we address the issue of Chinese Verdicts. 

💪 **Our goal is to identify the features of each verdict**, enabling us to **quickly find judgments with similar characteristics** during subsequent analysis. 

This allows us to perform simplified clustering or modeling to determine the judgment category. Additionally, by leveraging the features of each judgment, we can assist claims adjusters in **rapidly screening similar judgments, facilitating efficient matching of cases.**

So we implement the two-stage method into the following structure:



## Results


| **Model**         |  **Precision** | **Recall** |  **F1** |
|:-----------------:|:--------------:|:----------:|:-------:|
|      Ernie        |  Completed  |  Eric   | 2022/10/1     |
|  Ernie (chunk)    | In Progress |  Eric   | 2022/9/30     |
|  XLNet            | Not statred |  Eric   | 2022/9/20     |
|Two-Stage (UIE+UTC)|             |         |               |


## Quick Start



