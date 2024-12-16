from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from typing import List

def rewrite_query(llm, query):

    print(f"Before re-write: {query}")
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )
    question_rewriter = re_write_prompt | llm | StrOutputParser()
    query = question_rewriter.invoke({"question": query})
    print(f"After re-write: {query}")
    return query

# 并查集的find和union操作
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            self.parent[rootY] = rootX

def merge_lists(lists: List[List[int]]) -> List[List[int]]:
    # 记录每个元素属于哪个列表
    element_to_index = {}
    n = len(lists)
    uf = UnionFind(n)
    
    # 遍历所有列表，记录元素所属的列表并合并有共同元素的列表
    for i, lst in enumerate(lists):
        for num in lst:
            if num in element_to_index:
                uf.union(i, element_to_index[num])
            else:
                element_to_index[num] = i

    # 根据并查集结果，合并列表
    merged = {}
    for i, lst in enumerate(lists):
        root = uf.find(i)
        if root not in merged:
            merged[root] = set()
        merged[root].update(lst)

    # 转化为List[List[int]]格式
    return [list(s) for s in merged.values()]

from nltk.stem import PorterStemmer
from collections import Counter
import regex
import string 
import pandas as pd 

ps = PorterStemmer()

def normalize_answer(s):

    s = s.replace(',', "")
    def remove_articles(text):
        # return regex.sub(r'\b(a|an|the)\b', ' ', text)
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    
    # if len(ground_truth_tokens) <= len(prediction_tokens) and all(token in prediction_tokens for token in ground_truth_tokens):
    #     precision = 1.0  # Consider it a match if all ground truth tokens are in the prediction
    
    
    f1 = (2 * precision * recall) / (precision + recall)
    # print('# F1 #', prediction, ' | ', ground_truth, ' #', precision, recall, f1)
    # return recall
    return f1


import numpy as np 

def f1(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(',')]
    ground_truths = [g.strip() for g in ground_truth.split(',')]
    # print('# F1 [multi-answer]#', predictions, ' | ', ground_truths, ' #', np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths]))
    return np.mean([max([f1_score(prediction, gt) for prediction in predictions]) for gt in ground_truths])

def exact_match_score(prediction, ground_truth):

    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    # print('# EM #', prediction, ' | ', ground_truth, ' #', set(prediction.split()) == set(ground_truth.split()))
    # return normalize_answer(prediction) == normalize_answer(ground_truth)
    return set(prediction.split()) == set(ground_truth.split())

def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

def rougel_score(prediction, ground_truth):
    from rouge import Rouge
    rouge = Rouge()
    prediction = ' '.join([ps.stem(w) for w in normalize_answer(prediction).split()])
    ground_truth = ' '.join([ps.stem(w) for w in normalize_answer(ground_truth).split()])
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-1"]["f"]


def rl(prediction, ground_truths):
    return max([rougel_score(prediction, gt) for gt in ground_truths])

def exact_match_score(prediction, ground_truth):
    return 1 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0


# import torch

# def to_device(inputs, device):

#     def dict_to_device(data):
#         return {k: item.to(device) if torch.is_tensor(item) else item for k, item in data.items()}
    
#     if isinstance(inputs, (tuple, list)):
#         new_data = [] 
#         for item in inputs:
#             if isinstance(item, dict):
#                 new_data.append(dict_to_device(item))
#             elif torch.is_tensor(item):
#                 new_data.append(item.to(device))
#             else:
#                 new_data.append(item)
#     elif isinstance(inputs, dict):
#         new_data =dict_to_device(inputs)
#     else:
#         raise TypeError(f"Currently do not support using <{type(inputs)}> as the type of a batch")

#     return new_data