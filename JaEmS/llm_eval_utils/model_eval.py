import os
import pandas as pd
import json
import numpy as np
import copy
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader

from tqdm.autonotebook import tqdm
from ..framework.JaEmSDataset import JaEmSDataset, collate_tokenize, TokDataset
from ..framework.framework_creation import create_eval_frame 
from ..load_utils.load_datasets import load_json


def load_pipeline(pipe_kwargs, eval_api=None):
    """
    Load the appropriate generation pipeline based on evaluation API.

    Parameters
    ----------
    pipe_kwargs : dict
        Arguments to configure the pipeline.
    eval_api : str or None
        Type of API to use for generation, e.g., 'vllm' or None for HF.

    Returns
    -------
    Callable
        A callable pipeline for text generation.
    """
    if eval_api == 'vllm':
        pipe = vllm_load_pipeline(pipe_kwargs)
    else:
        pipe = hf_load_pipeline(pipe_kwargs)
    return pipe

    
def hf_pipeline(batch_inputs, hf_pipe):
    """
    Run a Hugging Face pipeline on a batch of inputs.

    Parameters
    ----------
    batch_inputs : list
        List of input data for the pipeline.
    hf_pipe : transformers.Pipeline
        Preloaded Hugging Face pipeline.

    Returns
    -------
    list
        Generated text outputs.
    """
    batch_outputs = hf_pipe(batch_inputs)
    return [output['generated_text'][-1]['content'] for output in batch_outputs] 
                                

def hf_load_pipeline(pipe_kwargs, load_in_8bit=True):
    """
    Load a Hugging Face text generation pipeline.

    Parameters
    ----------
    pipe_kwargs : dict
        Arguments for configuring the pipeline.
    load_in_8bit : bool
        Whether to load the model in 8-bit precision.

    Returns
    -------
    Callable
        A wrapped pipeline function for generation.
    """
    if load_in_8bit:
        pipe_kwargs['model_kwargs'] = {"torch_dtype": torch.bfloat16, 
                       'quantization_config': BitsAndBytesConfig(load_in_8bit=load_in_8bit)}
    hf_pipe = pipeline(**pipe_kwargs)
    return lambda x: hf_pipeline(x[0], hf_pipe)
    

def vllm_pipeline(batch_text, model, tokenizer):
    """
    Run a vLLM generation pipeline.

    Parameters
    ----------
    batch_text : list
        Batched conversation inputs.
    model : Callable
        vLLM model's generate function.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for formatting input.

    Returns
    -------
    list
        Generated text outputs.
    """
    batch_input = [tokenizer.apply_chat_template(messages, 
                                                 add_generation_prompt=True)
                   for messages in batch_text]
    batch_output = model(batch_input)
    return [output.outputs[0].text for output in batch_output]


def vllm_load_pipeline(pipe_kwargs):
    """
    Load a vLLM generation pipeline.

    Parameters
    ----------
    pipe_kwargs : dict
        Configuration for the vLLM model and tokenizer.

    Returns
    -------
    Callable
        A wrapped vLLM generation function.
    """
    tokenizer = AutoTokenizer.from_pretrained(pipe_kwargs['model'])
    sampling_params = SamplingParams(temperature=1, 
                                     max_tokens=pipe_kwargs['max_new_tokens'],
                                     stop_token_ids=[tokenizer.eos_token_id])
    llm = LLM(model=pipe_kwargs['model'], trust_remote_code=True,
              enable_chunked_prefill=False, # https://github.com/vllm-project/vllm/issues/8024
              # enforce_eager=True,
              max_num_seqs=2,
              max_num_batched_tokens=3000,
              max_model_len=4096,
              gpu_memory_utilization=0.95, 
              tensor_parallel_size=2
             )
    llm_gen = lambda x: llm.generate(prompt_token_ids=x, sampling_params=sampling_params)
    return lambda x: vllm_pipeline(x, llm_gen, tokenizer)


def load_and_evaluate(eval_data_path):
    """
    Load evaluation data and run the evaluation process.

    Parameters
    ----------
    eval_data_path : str
        Path to the evaluation configuration JSON.
    """
    eval_data = load_json(eval_data_path)
    data_kwargs = eval_data['framework_data']    
    tokens = load_json(data_kwargs['acess_tokens'])
    eval_filling = load_json(data_kwargs['eval_filling'])
    n_tuples = load_json(data_kwargs['n_tuples'])
    prompt = data_kwargs['prompt']
    prompt_func = lambda x: {'role': prompt['role'], 
                             'content': prompt['content'].format(x)} 
    dataset_path = data_kwargs['eval_dataset']
    if data_kwargs.get('load_toks', False):
        eval_dataset = TokDataset(load_json(dataset_path))
    else:    
        if np.__version__[:2] == "1.":
            import sys
            sys.modules["numpy._core.numeric"] = np.core.numeric
        eval_frame = pd.read_pickle(dataset_path)
        eval_dataset = JaEmSDataset(eval_filling, eval_frame, n_tuples, 
                                    prompt=prompt_func)
    
    eval_dataloader = DataLoader(eval_dataset, collate_fn=collate_tokenize, 
                                 **eval_data['eval_kwargs']['dataloader'])
    pipe_kwargs = copy.deepcopy(eval_data['pipeline_kwargs'])
    pipe_kwargs['token'] = tokens['HF_API_KEY']
       
    eval_api = data_kwargs.get('eval_api', None)
    model = load_pipeline(pipe_kwargs, eval_api)
    eval_model_dims(eval_dataloader, model, 
                    batch_size=eval_data['eval_kwargs']['dataloader']['batch_size'],
                    **eval_data['eval_kwargs']['eval'])
    

def eval_model_dims(eval_dataloader,  model_pipe,
                    results_dir='', save_after_n=np.inf,
                    batch_size=2, initial_n=0):
    """
    Run evaluation over all dimensions and optionally save results.

    Parameters
    ----------
    eval_dataloader : DataLoader
        Dataloader with evaluation samples.
    model_pipe : Callable
        Generation pipeline callable.
    results_dir : str
        Directory to save results.
    save_after_n : int
        Interval for saving intermediate outputs.
    batch_size : int
        Batch size for evaluation.
    initial_n : int
        Initial batch index for naming saved files.

    Returns
    -------
    tuple
        Dictionary of generated conversations and the evaluation frame.
    """
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    conv_gens = {}
    curr_save_n = initial_n
    
    for i, (batch, n_tuples) in enumerate(tqdm(eval_dataloader, initial=initial_n)):
        output = model_pipe(batch)
        curr_gens = {str(i*batch_size + x): b for x, b in enumerate(output)}
        conv_gens = dict(conv_gens, **curr_gens)
        if len(conv_gens.keys()) >= save_after_n:
            with open(results_dir + f'/test_batch_{curr_save_n}.json', 'w') as f:
                json.dump(conv_gens, f)
            
            curr_save_n += len(conv_gens.keys())
            conv_gens = {}
        output=[]
        torch.cuda.empty_cache()
            
    with open(results_dir + f'/test_batch_{curr_save_n + 1}.json', 'w') as f:
        json.dump(conv_gens, f)
    # eval_frame.to_pickle(results_dir + '/eval_frame.pkl')
