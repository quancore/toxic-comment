import random
import re
import gc
import numpy as np
import pandas as pd
from nltk import sent_tokenize
from tqdm import tqdm
import albumentations
from albumentations.core.transforms_interface import BasicTransform
from pandarallel import pandarallel
import data_cleaning as cleaning

class NLPTransform(BasicTransform):
    """ Transform for nlp task."""
    LANGS = {
        'en': 'english',
        'it': 'italian', 
        'fr': 'french', 
        'es': 'spanish',
        'tr': 'turkish', 
        'ru': 'russian',
        'pt': 'portuguese'
    }

    @property
    def targets(self):
        return {"data": self.apply}
    
    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params

    def get_sentences(self, text, lang='en'):
        return sent_tokenize(str(text), self.LANGS.get(lang, 'english'))

class ShuffleSentencesTransform(NLPTransform):
    """ Do shuffle by sentence """
    def __init__(self, always_apply=False, p=0.5):
        super(ShuffleSentencesTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = str(text)
        sentences = self.get_sentences(text, lang)
        random.shuffle(sentences)
        return ' '.join(sentences), lang

class ExcludeDuplicateSentencesTransform(NLPTransform):
    """ Exclude equal sentences """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeDuplicateSentencesTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        sentences = set()
        for sentence in self.get_sentences(text, lang):
            sentence = sentence.strip()
            sentences.add(sentence)
        
        return ' '.join(sentences), lang

class ExcludeNumbersTransform(NLPTransform):
    """ exclude any numbers """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeNumbersTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = str(text)
        try:
            text = re.sub(r'[0-9]', '', text)
            text = re.sub(r'\s+', ' ', text)
        except:
            print(f'text: {text} | lang:{lang}')
        
        return text, lang

class ExcludeUsersMentionedTransform(NLPTransform):
    """ Exclude @users """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeUsersMentionedTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = str(text)
        try:
            text = re.sub(r'@[\S]+\b', '', text)
            text = re.sub(r'\s+', ' ', text)
        except:
            print(f'text: {text} | lang:{lang}')
        
        return text, lang

class ExcludeHashtagsTransform(NLPTransform):
    """ Exclude any hashtags with # """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeHashtagsTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = str(text)
        try:
            text = re.sub(r'#[\S]+\b', '', text)
            text = re.sub(r'\s+', ' ', text)
        except:
            print(f'text: {text} | lang:{lang}')
        
        return text, lang

class ExcludeUrlsTransform(NLPTransform):
    """ Exclude urls """
    def __init__(self, always_apply=False, p=0.5):
        super(ExcludeUrlsTransform, self).__init__(always_apply, p)

    def apply(self, data, **params):
        text, lang = data
        text = str(text)
        try:
            text = re.sub(r'https?\S+', '', text)
            text = re.sub(r'\s+', ' ', text)
        except:
            print(f'text: {text} | lang:{lang}')
        
        return text, lang

class SwapWordsTransform(NLPTransform):
    """ Swap words next to each other """
    def __init__(self, swap_distance=1, swap_probability=0.1, always_apply=False, p=0.5):
        """  
        swap_distance - distance for swapping words
        swap_probability - probability of swapping for one word
        """
        super(SwapWordsTransform, self).__init__(always_apply, p)
        self.swap_distance = swap_distance
        self.swap_probability = swap_probability
        self.swap_range_list = list(range(1, swap_distance+1))

    def apply(self, data, **params):
        text, lang = data
        words = text.split()
        words_count = len(words)
        if words_count <= 1:
            return text, lang

        new_words = {}
        for i in range(words_count):
            if random.random() > self.swap_probability:
                new_words[i] = words[i]
                continue
    
            if i < self.swap_distance:
                new_words[i] = words[i]
                continue
    
            swap_idx = i - random.choice(self.swap_range_list)
            new_words[i] = new_words[swap_idx]
            new_words[swap_idx] = words[i]

        return ' '.join([v for k, v in sorted(new_words.items(), key=lambda x: x[0])]), lang


class CutOutWordsTransform(NLPTransform):
    """ Remove random words """
    def __init__(self, cutout_probability=0.05, always_apply=False, p=0.5):
        super(CutOutWordsTransform, self).__init__(always_apply, p)
        self.cutout_probability = cutout_probability

    def apply(self, data, **params):
        text, lang = data
        words = text.split()
        words_count = len(words)
        if words_count <= 1:
            return text, lang
        
        new_words = []
        for i in range(words_count):
            if random.random() < self.cutout_probability:
                continue
            new_words.append(words[i])

        if len(new_words) == 0:
            return words[random.randint(0, words_count-1)], lang

        return ' '.join(new_words), lang

class AddNonToxicSentencesTransform(NLPTransform):
    """ Add random non toxic statement """
    def __init__(self, non_toxic_sentences, sentence_range=(1, 3), always_apply=False, p=0.5):
        super(AddNonToxicSentencesTransform, self).__init__(always_apply, p)
        self.sentence_range = sentence_range
        self.non_toxic_sentences = non_toxic_sentences

    def apply(self, data, **params):
        text, lang = data

        sentences = self.get_sentences(text, lang)
        for i in range(random.randint(*self.sentence_range)):
            sentences.append(random.choice(self.non_toxic_sentences))
        
        random.shuffle(sentences)
        return ' '.join(sentences), lang

class SynthesicOpenSubtitlesTransform(NLPTransform):
    def __init__(self, path, always_apply=False, p=0.5):
        super(SynthesicOpenSubtitlesTransform, self).__init__(always_apply, p)
        df = pd.read_csv(path, index_col='id')[['comment_text', 'toxic', 'lang']]
        df = df[~df['comment_text'].isna()]
        df = cleaning.clean_data(df, ['comment_text'])
        df = df.drop_duplicates(subset='comment_text')
        df['toxic'] = df['toxic'].round().astype(np.int)

        self.synthesic_toxic = df[df['toxic'] == 1].comment_text.values
        self.synthesic_non_toxic = df[df['toxic'] == 0].comment_text.values

        del df
        gc.collect();

    def generate_synthesic_sample(self, text, toxic):
        try:
            if isinstance(text, (np.ndarray, np.generic)):
                text = text[0]

            texts = [text]
            if toxic == 0:
                for i in range(random.randint(1,5)):
                    texts.append(random.choice(self.synthesic_non_toxic))
            else:
                for i in range(random.randint(0,2)):
                    texts.append(random.choice(self.synthesic_non_toxic))
                
                for i in range(random.randint(1,3)):
                    texts.append(random.choice(self.synthesic_toxic))
            
            random.shuffle(texts)
            return ' '.join(texts)
        except:
            print(f'text: {text}')
            print(f'texts: {texts}')
            raise
        

    def apply(self, data, **params):
        text, toxic = data
        text = self.generate_synthesic_sample(text, toxic)
        return text, toxic



def get_non_toxic(non_toxic_path):
    nlp_transform = NLPTransform()
    df = pd.read_csv(non_toxic_path, nrows=1000)
    df = df[df.toxic == 0]
    df['lang'] = 'en'
    non_toxic_sentences = set()
    for comment_text in tqdm(df['comment_text'], total=df.shape[0]):
        non_toxic_sentences.update(nlp_transform.get_sentences(comment_text), 'en')
    
    return list(non_toxic_sentences)

def translate_transformation(name, p, open_subtitles_path=None, non_toxic_path=None):
    if name == 'nontoxic':
        if non_toxic_path == None:
            raise ValueError('Non toxic path is None')
        
        return AddNonToxicSentencesTransform(non_toxic_sentences=get_non_toxic(non_toxic_path), p=p, sentence_range=(1,3)),
    
    elif name == 'shuffle':
        return ShuffleSentencesTransform(p=p)
    
    elif name == 'swap':
        return SwapWordsTransform(p=p)
    
    elif name == 'cutout':
        return CutOutWordsTransform(p=p)
    
    elif name == 'synthesic':
        if open_subtitles_path == None:
            raise ValueError('Open subtitles path is None')
        return SynthesicOpenSubtitlesTransform(open_subtitles_path, p=p)
    
    elif name == 'exc_num':
        return ExcludeNumbersTransform(p=p)
    
    elif name == 'exc_hashtag':
        return ExcludeHashtagsTransform(p=p)
    
    elif name == 'exc_duplicate':
        return ExcludeDuplicateSentencesTransform(p=p)
    
    elif name == 'exc_url':
        return ExcludeUrlsTransform(p=p)
    
    elif name == 'exc_mention':
        return ExcludeUsersMentionedTransform(p=p)
    
def get_transforms(transformations, non_toxic_path=None, open_subtitles_path=None):
    if len(transformations) == 1:
        return translate_transformation(transformations[0][0], transformations[0][1], non_toxic_path=non_toxic_path, open_subtitles_path=open_subtitles_path)
    
    transf_f = []
    for trans in transformations:
        trans_type = trans[0]
        if trans_type == 'oneof':
            transform_types = trans[1]
            step = albumentations.OneOf([translate_transformation(trans_t, prob, non_toxic_path=non_toxic_path) 
                                         for trans_t, prob in transform_types])
        
        else:
            p = trans[1]
            step = translate_transformation(trans_type, p, non_toxic_path=non_toxic_path, open_subtitles_path=open_subtitles_path)
        
        transf_f.append(step)
    
    return albumentations.Compose(transf_f)
