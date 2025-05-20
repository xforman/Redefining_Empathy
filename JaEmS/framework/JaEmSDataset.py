from torch.utils.data import Dataset


class JaEmSDataset(Dataset):
    def __init__(self, eval_data, eval_frame, n_tuples, prompt=None):
        self.frame = eval_frame 
        self.data = eval_data
        # expects the prompt to be a function taking one arg or is None
        self.prompt = prompt 
        self.n_tuples = n_tuples 

    
    def __len__(self):
        return self.frame.shape[0]

    
    def __getitem__(self, idx):
        sf = self.frame.loc[idx] # sample frame protected attributes
        select = sf['attr_selection']
        test_sample = sf['test_samples']
        n_tuple = self.n_tuples[test_sample]['n-tuple']
        input_conv = []
        
        for i, soc_grp in enumerate(sf['dim_order']):
            conv_sel = select[i]
            conv = self.data[soc_grp][sf[soc_grp]]['conversations'][conv_sel]['text']
            input_conv += conv
        if self.prompt is not None:
            input_conv.append(self.prompt(n_tuple[-1]))
        return input_conv, n_tuple


def collate_tokenize(data):
    batch, nTuples = zip(*data)
    return batch, nTuples


class TokDataset(Dataset):
    def __init__(self, toks):
        self.toks = toks 
    
    def __len__(self):
        return len(self.toks)

    def __getitem__(self, idx):
        toks = self.toks[idx]
        return toks, []
        