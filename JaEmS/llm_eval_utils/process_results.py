import json
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .model_eval import load_json
from .metrics import self_consistency, baseline_consistency

from typing import List


def load_eval_batches(eval_batches_path, 
                      batch_name_template='/test_batch_*.json'):
    """
    Load all evaluation batch result files from the given directory.

    Args:
        eval_batches_path (str): Path to the evaluation results directory.
        batch_name_template (str): File pattern to match evaluation batch files.

    Returns:
        dict: A dictionary mapping example IDs to generated outputs.
    """
    eval_dict = {}
    batch_path = eval_batches_path + batch_name_template
    for x in glob.glob(batch_path):
        batch_dict = load_json(x)
        eval_dict  = dict(eval_dict, **batch_dict)
    return eval_dict
    

def load_all_results(eval_metadata_path):
    """
    Load the entire evaluation results including model generations and human annotations.

    Args:
        eval_metadata_path (str): Path to the JSON file containing evaluation metadata.

    Returns:
        tuple: (result_frame: pd.DataFrame, human_cot: pd.DataFrame)
            - result_frame contains merged model outputs and metadata.
            - human_cot contains human chain-of-thought annotations.
    """
    eval_metadata = load_json(eval_metadata_path)
    framework_data = eval_metadata['framework_data']
    
    eval_frame = pd.read_pickle(framework_data['eval_dataset'])
    n_tuples = load_json(framework_data['n_tuples'])
    human_cot = pd.DataFrame(n_tuples.values(), index=n_tuples.keys(),
                             columns=['human_cot'])
    
    batch_size = eval_metadata['eval_kwargs']['dataloader']['batch_size']
    eval_dir  = eval_metadata['eval_kwargs']['eval']['results_dir'] 
    eval_dict = load_eval_batches(eval_dir)
    gens = pd.DataFrame(eval_dict.values(), index=eval_dict.keys(), 
                        columns=['gens'])
    gens.index = gens.index.astype('int')
    
    result_frame = pd.merge(eval_frame, gens, left_index=True, right_index=True)
    #result_frame = result_frame.merge(human_cot, left_on='test_samples', 
    #                                  right_index=True)
    return result_frame, human_cot
    

def split_col_on_token(gen, token='\n', remove_first_n=3, split_kwargs={}):
    """
    Split text column on a token and optionally remove the first N characters of each part.

    Args:
        gen (pd.Series): A Series of strings to be split.
        token (str): The token to split on.
        remove_first_n (int): Number of characters to remove from each split string.
        split_kwargs (dict): Additional keyword arguments for `str.split`.

    Returns:
        pd.DataFrame: A DataFrame of split strings.
    """
    return (gen.str.replace('\n\n', '\n')
               .str.split(token, expand=True, **split_kwargs)
               .apply(lambda x: x.str[remove_first_n:]))

    
def process_results(result_frame, col_to_embed, 
                    cot_split_func, embedding_model=None):
    """
    Process a DataFrame to split CoT (chain-of-thought) steps and optionally embed them.

    Args:
        result_frame (pd.DataFrame): The main DataFrame containing model outputs.
        col_to_embed (str): The column in the DataFrame to be processed.
        cot_split_func (Callable): Function to split CoT outputs into steps.
        embedding_model (Callable, optional): Embedding function to apply on CoT steps.

    Returns:
        tuple: (processed_frame: pd.DataFrame, embeddings: np.ndarray or list)
    """
    # eval_frame with all of the outputs in one columns named 'gens'
    gens = result_frame[col_to_embed]
    
    cot_gens = cot_split_func(gens)
    print(cot_gens.columns[:-1])
    gens = pd.melt(cot_gens.reset_index(), id_vars='index',
                  value_vars=cot_gens.columns, 
                   var_name='cot_step', value_name='cot_gens')
    gens['cot_step'] = gens['cot_step'].max() + 1 - gens['cot_step']
    
    # replace None with something the embedding model can eat
    split_result_frame = pd.merge(result_frame, gens, left_index=True,
                                  right_on='index') 
    split_result_frame = split_result_frame.reset_index(drop=True)
    split_result_frame.rename(columns={'index': 'example_index'}, inplace=True)
    
    if embedding_model is not None:
        to_emb = split_result_frame['cot_gens'].fillna('')
        embs = embedding_model(to_emb.to_numpy())
    else:
        embs = []
    return split_result_frame, embs


def compute_consistency(x, consistency, similarity, embs, 
                        baseline_dims, baseline):
    """
    Compute consistency scores for a group of examples.

    Args:
        x (pd.DataFrame): Subgroup of data for which consistency is computed.
        consistency (Callable): Consistency function (e.g., self or baseline).
        similarity (Callable): Similarity function to compute embedding similarity.
        embs (np.ndarray): Array of embeddings for all examples.
        baseline_dims (list): Dimensions to match for baseline.
        baseline (pd.DataFrame): Data used as the baseline.

    Returns:
        pd.DataFrame: A DataFrame of consistency scores indexed by example index.
    """
    vals = baseline.reset_index().merge(x[baseline_dims].head(1),
                                        on=baseline_dims)
    S = embs[vals['index']]
    s = embs[x.index]
    consists = consistency(s, S, similarity)
    consts_group = pd.DataFrame(consists, columns=['XY_consistency'], index=x.index) 
    consts_group = consts_group.reset_index()
    return consts_group
    

def get_consistency(cot_frame, embs, dims: List[str], kind: str, similarity, 
                    baseline_dims: List[str]=None, baseline=None):
    """
    Compute self- or baseline-consistency over grouped CoT outputs.

    Args:
        cot_frame (pd.DataFrame): Processed CoT result DataFrame.
        embs (np.ndarray): Embeddings of CoT steps.
        dims (List[str]): Dimensions to group by (e.g., question, label).
        kind (str): Consistency type: 'self' or 'baseline'.
        similarity (Callable): Similarity function for embedding comparison.
        baseline_dims (List[str], optional): Dimensions for baseline comparison.
        baseline (pd.DataFrame, optional): DataFrame to use as baseline group.

    Returns:
        pd.DataFrame: DataFrame with consistency scores merged into original CoT frame.
    """
    # https://stackoverflow.com/questions/77969964/
    # deprecation-warning-with-groupby-apply
    select_groups = cot_frame.groupby(dims, sort=False)[cot_frame.columns]
    # create baseline groups
    consistency = self_consistency if kind == 'self' else baseline_consistency
    
    if baseline is None or baseline_dims is None:
        baseline = cot_frame
        baseline_dims = dims
        
    consists = select_groups.apply(compute_consistency,
                               consistency=consistency,
                               similarity=similarity, embs=embs,
                               baseline_dims=baseline_dims, baseline=baseline)
    
    consist_col_name = kind + '-consistency_OF_' + '_X_'.join(dims)
    consists = consists.rename(columns={'XY_consistency': consist_col_name})
    consists = consists.reset_index(drop=True)
    
    consists_frame = pd.merge(cot_frame, consists,
                              right_on=['index'], left_index=True)
    consists_frame = consists_frame.set_index('index')
    return consists_frame
