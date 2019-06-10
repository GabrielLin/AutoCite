from typing import Dict
import pandas as pd
import numpy as np

#move calls to nnselect and nnrank into helper functions so that eval_text_model can be used for alternative architectures:
def get_candidates(**args):
    Pass
    
def rank_candidates(**args):
    Pass

def gold_citations(doc_id: int, df: pd.DataFrame, min_citations: int, id_dict: Dict):
    cite_ids = df.iloc[doc_id].outCitations
    gold_citations_1 = set(filter(None,[id_dict.get(i) for i in cite_ids]))
    
    if doc_id in gold_citations_1:
        gold_citations_1.remove(doc_id)
        
    citations_of_citations = []
    for c in gold_citations_1:
        cite_ids = df.iloc[c].outCitations
        citations_of_citations.extend(list(filter(None,[id_dict.get(i) for i in cite_ids])))
        
    gold_citations_2 = set(citations_of_citations).union(gold_citations_1)
    
    if doc_id in gold_citations_2:
        gold_citations_2.remove(doc_id)
        
    if len(gold_citations_1) < min_citations:
        return [], []

    return gold_citations_1, gold_citations_2
