import pandas as pd
import numpy as np
import copy


def create_frame(dataset: dict[dict], dims, random=False, n_samples=None):
    """
    Generate a pandas DataFrame representing a multi-dimensional sampling frame from a dataset.

    Parameters
    ----------
    dataset : dict of dict
        A nested dictionary where keys represent social groups (dimensions) and subkeys represent attributes.
    dims : list of str
        A list of keys corresponding to the dimensions to include in the frame.
    random : bool, optional
        If `True`, generates a random sample of the Cartesian product. Default is `False`.
    n_samples : int or None, optional
        Number of samples to generate. If `None`, the full Cartesian product is used.

    Returns
    -------
    frame : pandas.DataFrame
        A DataFrame of samples with each column corresponding to a selected dimension.
    """ 
    dim_sizes = np.array([len(dataset[dim].keys()) for dim in dims])
    enc_dims = [np.arange(n_dim) for n_dim in dim_sizes]
    n_dims = dim_sizes.shape[0]
    if n_samples is None:
        n_samples = dim_sizes.prod()
    
    if random:
        frame = np.random.randint(n_samples, size=(n_samples, n_dims))
        frame = frame % dim_sizes
    else: # this way the columns start each at a multiple of n_samples
        #frame = np.arange(n_dims*n_samples).reshape(n_dims, n_samples).T
        #frame = frame // (n_samples/np.cumprod(dim_sizes)) 
        frame = np.array(np.meshgrid(*enc_dims)).T.reshape(-1, n_dims)
       
    return pd.DataFrame(frame, columns=dims)  


def shuffle_dims(dims, n_samples):
    """
    Generate multiple random permutations of a dimension order.

    Parameters
    ----------
    dims : list
        A list of dimension labels.
    n_samples : int
        Number of random permutations to generate.

    Returns
    -------
    shuffled : list of list
        A list containing `n_samples` shuffled versions of `dims`.
    """    
    return [np.random.permutation(dims) for _ in range(n_samples)]


def select_dims(dataset, to_remove=[], add_empty=[], inplace=False):
    """
    Modify the dataset by adding empty dimensions or removing specific ones.

    Parameters
    ----------
    dataset : dict of dict
        The dataset to modify.
    to_remove : list of str, optional
        Dimensions to overwrite with empty attributes.
    add_empty : list of str, optional
        Dimensions to add with a default empty attribute.
    inplace : bool, optional
        If `True`, modify the dataset in place. Otherwise, return a new modified copy.

    Returns
    -------
    new_dataset : dict of dict
        The modified dataset.
    """
    new_dataset = copy.deepcopy(dataset) if not inplace else dataset
    empty_attr = {'value': '', 'conversations': {'-': {'text': [], 'utils': '',
                                'creation_info': 'created as an option to exclude a social group' }}}
    for dim in add_empty:
        new_dataset[dim]['-'] = copy.deepcopy(empty_attr)
    for dim in to_remove:
        new_dataset[dim] = {'-': copy.deepcopy(empty_attr)}
    
    return new_dataset
    

def conv_selection(x, dataset, i2attr, i2conv, ignore_cols=[]):
    """
    Select conversation instances from the dataset based on a row's dimension ordering and attribute indices.

    Parameters
    ----------
    x : pandas.Series
        A row from the evaluation frame containing dimension values and their order.
    dataset : dict
        The dataset containing conversations grouped by attributes and dimensions.
    i2attr : dict
        A mapping from dimension names to a dictionary mapping indices to attribute names.
    i2conv : dict
        A mapping from dimension and attribute to available conversation IDs.
    ignore_cols : list, optional
        List of column names to ignore during selection.

    Returns
    -------
    selection : list
        A list of selected conversation IDs, one for each dimension in `dim_order`.
    """
    selection = [[] for _ in x['dim_order']]
    conv_count = {sg: 0 for sg in dataset.keys()}
    for i, soc_grp in reversed(list(enumerate(x['dim_order']))):
        pa = i2attr[soc_grp][x[soc_grp]]
        convs = dataset[soc_grp][pa]
        max_conv_n =  len(convs['conversations'].keys()) - 1
        
        if soc_grp not in ignore_cols:
            conv_count[soc_grp] = np.random.randint(
                conv_count[soc_grp], max_conv_n + 1)  

        selection[i] = i2conv[soc_grp][pa][max_conv_n - conv_count[soc_grp]]
        conv_count[soc_grp] += 1
    return selection
    

def create_eval_frame(dataset, eval_dims: list[str], dim_order: list[str], 
                      bootstrap=5, repetition=True, shuffle_dim_order=True):
    """
    Generate an evaluation frame by combining deterministic sampling over evaluation dimensions
    and random sampling over the rest.

    Parameters
    ----------
    dataset : dict
        The dataset containing nested dictionaries for each dimension and attribute.
    eval_dims : list of str
        Dimensions to include in the Cartesian product.
    dim_order : list of str
        Base dimension ordering for evaluation. Can be shuffled if enabled.
    bootstrap : int, optional
        Number of repetitions for each sample when `repetition` is `False`. Default is 5.
    repetition : bool, optional
        If `True`, each sample is repeated `bootstrap` times. If `False`, product is bootstrapped.
    shuffle_dim_order : bool, optional
        Whether to shuffle the order of dimensions for each sample. Default is `True`.

    Returns
    -------
    eval_frame : pandas.DataFrame
        A DataFrame where each row represents a sample and its attribute selections.
    i2attr : dict
        A mapping from dimensions to index-to-attribute dictionaries.
    """
    # create the Cartesian product of attributes soc. groups from evalDims
    prod_frame = create_frame(dataset, eval_dims)
    # randomly sample attributes from the rest of the dimensions
    n_samples = prod_frame.shape[0] * (bootstrap if not repetition else 1) 
    rand_dims = [soc_grp for soc_grp in dataset.keys() if soc_grp not in eval_dims]
    rand_frame = create_frame(dataset, rand_dims, True, n_samples)
    
    prod_frame['dim_order'] = dim_order if not shuffle_dim_order else \
                                shuffle_dims(dim_order, n_samples) 
    # label the attributes
    i2attr = {soc_grp: {i: pa for i, pa in enumerate(pas.keys())}
              for soc_grp, pas in dataset.items()}
    i2conv = {sg: {pa: list(convs['conversations'].keys()) 
                   for pa, convs in pas.items()} 
              for sg, pas in dataset.items()}
    
    cs_kwargs = {'dataset': dataset, 'i2attr': i2attr, 'ignore_cols': 
                 ['test_samples'], 'i2conv': i2conv}
    if repetition:
        eval_frame = pd.concat([prod_frame, rand_frame], axis=1)
        eval_frame['attr_selection'] = eval_frame.apply(conv_selection, 
                                                        axis=1, **cs_kwargs) 
        eval_frame = pd.concat([eval_frame]*bootstrap)    
    else:
        prod_frame = pd.concat([prod_frame]*bootstrap)       
        eval_frame = pd.concat([prod_frame, rand_frame], axis=1)
        eval_frame['attr_selection'] = eval_frame.apply(conv_selection,
                                                        axis=1, **cs_kwargs) 
    eval_frame = eval_frame.reset_index().rename(columns={'index':'sample_index'})
    for soc_grp, repl_w in i2attr.items():
        eval_frame[soc_grp] = eval_frame[soc_grp].replace(repl_w)
    return eval_frame, i2attr