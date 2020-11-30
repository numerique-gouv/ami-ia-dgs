import numpy as np
from tqdm import tqdm
from text_model_boris.utils import misc

from sklearn.preprocessing import OneHotEncoder

from transformers import BertTokenizer, CamembertTokenizer
from torch.utils.data import Dataset

tokenizers = {
    'bert-base-uncased': BertTokenizer,
    'camembert/camembert-base-ccnet': CamembertTokenizer
}


def tokenize(df, pretrained_model_str='bert-base-uncased'):
    print(f'Tokenize inputs for model {pretrained_model_str}...')

    tokenizer = tokenizers[pretrained_model_str].from_pretrained(pretrained_model_str)
    seg_ids_all, ids_all, attn_all = {}, {}, {}
    max_seq_len = 512
    
    for text, cols in [('libelle_description', [misc.input_columns[0], misc.input_columns[1]] ), 
                       ('libelle_etat', [misc.input_columns[0], misc.input_columns[2]] )]:
        ids, seg_ids, attn = [], [], []
        for x1, x2 in tqdm(df[cols].values):
            encoded_inputs = tokenizer.encode_plus(
                x1, x2, add_special_tokens=True, max_length=max_seq_len, 
                pad_to_max_length=True, return_token_type_ids=True
            )
            ids.append(encoded_inputs['input_ids'])
            seg_ids.append(encoded_inputs['token_type_ids'])
            attn.append(encoded_inputs['attention_mask'])
        
        ids_all[text] = np.array(ids)
        seg_ids_all[text] = np.array(seg_ids)
        attn_all[text] = np.array(attn)
    
    return ids_all, seg_ids_all, attn_all

def get_ohe_categorical_features(train, test, feature='FABRICANT'):
    unique_vals = list(set(train[feature].unique().tolist() 
                           + test[feature].unique().tolist()))
    feat_dict = {i + 1: e for i, e in enumerate(unique_vals)}
    feat_dict_reverse = {v: k for k, v in feat_dict.items()}

    train_feat = train[feature].apply(lambda x: feat_dict_reverse[x]).values.reshape(-1, 1)
    test_feat = test[feature].apply(lambda x: feat_dict_reverse[x]).values.reshape(-1, 1)

    ohe = OneHotEncoder(handle_unknown = 'ignore')
    ohe.fit(train_feat)
    train_feat = ohe.transform(train_feat).toarray()
    test_feat = ohe.transform(test_feat).toarray()

    print(train[feature].nunique())

    return train_feat, test_feat


class TextDataset(Dataset):

    def __init__(self, x_features, description_ids, etat_ids, seg_description_ids, 
                 seg_etat_ids, description_attn, etat_attn, idxs, targets=None):
        self.description_ids = description_ids[idxs].astype(np.long)
        self.etat_ids = etat_ids[idxs].astype(np.long)
        self.seg_description_ids = seg_description_ids[idxs].astype(np.long)
        self.seg_etat_ids = seg_etat_ids[idxs].astype(np.long)
        self.description_attn = description_attn[idxs].astype(np.long)
        self.etat_attn = etat_attn[idxs].astype(np.long)
        self.x_features = x_features[idxs].astype(np.float32)
        if targets is not None: self.targets = targets[idxs].astype(np.float32)
        else: self.targets = np.zeros((self.x_features.shape[0], len(misc.target_columns)), dtype=np.float32)

    def __getitem__(self, idx):
        d_ids = self.description_ids[idx]
        e_ids = self.etat_ids[idx]
        seg_d_ids = self.seg_description_ids[idx]
        seg_e_ids = self.seg_etat_ids[idx]
        attn_d = self.description_attn[idx]
        attn_e = self.etat_attn[idx]
        x_feats = self.x_features[idx]
        target = self.targets[idx]
        return x_feats, d_ids, e_ids, seg_d_ids, seg_e_ids, attn_d, attn_e, target

    def __len__(self):
        return len(self.x_features)


    
def get_pseudo_set(args, pseudo_df, tokenizer):
    return tokenize(pseudo_df, tokenizer)


def get_test_set(args, test_df, tokenizer):
    return tokenize(test_df, tokenizer)
