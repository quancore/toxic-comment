import os
import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# data related functions
def read_data(dir_paths, languages):
    '''
    Read the related data and return as dataframes.
    '''
    train = pd.read_csv((dir_paths['base_dir']/dir_paths['train_file1']))
    
    if 'train_file2' in dir_paths:
        train2 = pd.read_csv((dir_paths['base_dir']/dir_paths['train_file2']))
        train2 = train2[(train2['toxic'] < 0.2) | (train2['toxic'] > 0.5)]
        train = train.append(train2)
    
    train['lang'] = 'en'
    
    for lang in languages:
        try:
            df_path = (dir_paths['base_t_dir']/f'jigsaw-toxic-comment-train-google-{lang}-cleaned.csv')
            df_lang = pd.read_csv(df_path)
        except:
            print(f'Translation has not found: {df_path}')
        else:
            df_lang['lang'] = lang
            train = train.append(df_lang)
    
    train = train[~train['comment_text'].isna()]
    train = train.drop_duplicates(subset='comment_text')
    train['toxic'] = train['toxic'].round().astype(np.int32)
    
    valid = pd.read_csv((dir_paths['base_dir']/dir_paths['val_file']), index_col='id')
    test = pd.read_csv((dir_paths['base_dir']/dir_paths['test_file']), index_col='id')
    sub = pd.read_csv((dir_paths['base_dir']/dir_paths['sub_file']))
    
    return train, valid, test, sub

def read_external_data(root_path, languages):
    '''
    Read any external data source and return as a single dataframe.
    '''
    DFs, return_df = [], None

    for lang in languages:
        try:
            df_path = (root_path/f'{lang}-external-cleaned.csv')
            if os.path.isfile(df_path):
                df_lang = pd.read_csv(df_path)
                DFs.append(df_lang)
            else:
                print(f'File not found: {df_path}')
        except Exception as e:
            print(f'Error on reading external file: {df_path}')
            raise
    
    if DFs:
        return_df = pd.concat(DFs, ignore_index=True)
    
    return return_df

# encoders
def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    Chunk encode given texts.
    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)

def regular_encode(texts, tokenizer, maxlen=512, return_mask=False, batch_encode=True):
    '''
    Normal encode given texts.
    '''
    if batch_encode:
        enc_di = tokenizer.batch_encode_plus(
            texts, 
            add_special_tokens=True, 
            pad_to_max_length=True,
            max_length=maxlen
        )
    else:
        enc_di = tokenizer.encode_plus(
            texts, 
            add_special_tokens=True, 
            pad_to_max_length=True,
            max_length=maxlen
        )
    
    if return_mask:
        return np.array(enc_di['input_ids']), np.array(enc_di['attention_mask'])
    
    return np.array(enc_di['input_ids'])