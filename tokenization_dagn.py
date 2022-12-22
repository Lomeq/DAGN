''' encoding=utf-8

Data processing Version 3.2

Date: 10/11/2020
Author: Yinya Huang

* argument words: pre-defined relations.
* domain words: repeated n-gram.


* relations patterns:
    1 - (relation, head, tail)  关键词在句首
    2 - (head, relation, tail)  关键词在句中，先因后果
    3 - (tail, relation, head)  关键词在句中，先果后因


== graph ==
    * edges: periods.
    * edges: argument words.
    * nodes: chunks split by periods & argument words.

'''

from dataclasses import dataclass, field
import argparse
from transformers import AutoTokenizer
import gensim
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer("english")   #将一个单词的各种形式转换为原型(例：jumping->jump)


def token_stem(token):
    return stemmer.stem(token)


def arg_tokenizer(text_a, text_b, tokenizer, stopwords, relations:dict, punctuations:list,
                  max_gram:int, max_length:int, do_lower_case:bool=False):
    '''
    :param text_a: str. (context in a sample.)
    :param text_b: str. ([#1] option in a sample. [#2] question + option in a sample.)
    :param tokenizer: RoBERTa tokenizer.
    :param relations: dict. {argument words: pattern}
    :return:
        - input_ids: list[int]
        - attention_mask: list[int]
        - segment_ids: list[int]
        - argument_bpe_ids: list[int]. value={ -1: padding,
                                                0: non_arg_non_dom,
                                                1: (relation, head, tail)  关键词在句首
                                                2: (head, relation, tail)  关键词在句中，先因后果
                                                3: (tail, relation, head)  关键词在句中，先果后因
                                                }
        - domain_bpe_ids: list[int]. value={ -1: padding,
                                              0:non_arg_non_dom,
                                           D_id: domain word ids.}
        - punctuation_bpe_ids: list[int]. value={ -1: padding,
                                                   0: non_punctuation,
                                                   1: punctuation}
    '''

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def _is_stopwords(word, stopwords):
        if word in stopwords:
            is_stopwords_flag = True
        else:
            is_stopwords_flag = False
        return is_stopwords_flag

    def _head_tail_is_stopwords(span, stopwords):    #首尾词是否有stopword
        if span[0] in stopwords or span[-1] in stopwords:
            return True
        else:
            return False

    def _with_septoken(ngram, tokenizer):   #表明ngram是否包含tokenizer插入的任何特殊标记
        if tokenizer.bos_token in ngram or tokenizer.sep_token in ngram or tokenizer.eos_token in ngram:
            flag = True
        else: flag = False
        return flag

    def _is_argument_words(seq:str, argument_words):  #返回seq这个字符串的逻辑连接词(argument_words)类型
        pattern = None
        arg_words = list(argument_words.keys())
        if seq.strip() in arg_words:    #strip()移除首位空格
            pattern = argument_words[seq.strip()]
        return pattern

    def _is_exist(exists:list, start:int, end:int):    #检查一个给定的范围是否存在于一个范围列表中
        flag = False
        for estart, eend in exists:
            if estart <= start and eend >= end:
                flag = True
                break
        return flag

    def _find_punct(tokens, punctuations):  #如果某个位置有标点符号，返回列表中的元素为1，如果没有标点符号，该元素将是0
        punct_ids = [0] * len(tokens)
        for i, token in enumerate(tokens):
            if token in punctuations:
                punct_ids[i] = 1
        return punct_ids
    
    #返回值：
    # argument_words(dict)：逻辑连接词：每个逻辑连接词在tokens中对应的起始位置(tuple)
    # argument_ids(list)：tokens列表大小，但是把逻辑连接词都替换为对应的pattern值(tuple)
    def _find_arg_ngrams(tokens, max_gram):   #找到长度为1~max_gram长度的逻辑连接词
        n_tokens = len(tokens)
        global_arg_start_end = []
        argument_words = {}
        argument_ids = [0] * n_tokens
        for n in range(max_gram, 0, -1):  # loop over n-gram(从max_gram开始每次都用n大小的窗口遍历所有token).
            for i in range(n_tokens - n):  # n-gram window sliding.
                window_start, window_end = i, i + n
                ngram = " ".join(tokens[window_start:window_end]) #"seperator".join(...)
                pattern = _is_argument_words(ngram, relations)
                if pattern:
                    if not _is_exist(global_arg_start_end, window_start, window_end):  #如果该逻辑连接词不在global_arg_start_end这个保存范围的列表中，就加进去
                        global_arg_start_end.append((window_start, window_end))
                        argument_ids[window_start:window_end] = [pattern] * (window_end - window_start)  #用该逻辑连接词对应的逻辑类型替换对应位置的0
                        argument_words[ngram] = (window_start, window_end)  #每个逻辑连接词在tokens中对应的起始位置

        return argument_words, argument_ids

    # 返回值：
    # domain_words_stemmed(dict)：domain_stemmed_words:每个domain word在stemmed_tokens中对应的起始位置(list)
    # domain_words_orin(dict)：domain_words:每个domain word在tokens中对应的起始位置(list)
    # domain_ids(list): tokens列表大小，但是把domain words的位置都替换为对应的d_id值(从0开始)
    def _find_dom_ngrams_2(tokens, max_gram):
        '''
        1. 先遍历一遍，记录不是stopwords/septoken的字符
        2. 遍历记录的 n-gram, 如果其中一个被另一个包含，就保留较长的那一个
        3. 赋值 domain_ids.

        '''

        stemmed_tokens = [token_stem(token) for token in tokens]

        ''' 1 '''
        n_tokens = len(tokens)
        d_ngram = {}    #不是stopwords/septoken的stemmed_ngram:对应的起始位置列表
        domain_words_stemmed = {}
        domain_words_orin = {}
        domain_ids = [0] * n_tokens
        for n in range(max_gram, 0, -1):  # loop over n-gram.
            for i in range(n_tokens - n):  # n-gram window sliding.
                
                window_start, window_end = i, i+n
                stemmed_span = stemmed_tokens[window_start:window_end]
                stemmed_ngram = " ".join(stemmed_span)
                orin_span = tokens[window_start:window_end]
                orin_ngram = " ".join(orin_span)

                if _is_stopwords(orin_ngram, stopwords): continue
                if _head_tail_is_stopwords(orin_span, stopwords): continue
                if _with_septoken(orin_ngram, tokenizer): continue

                if not stemmed_ngram in d_ngram:
                    d_ngram[stemmed_ngram] = []
                d_ngram[stemmed_ngram].append((window_start, window_end))

        ''' 2 '''
        d_ngram = dict(filter(lambda e: len(e[1]) > 1, d_ngram.items()))  #保留出现至少两次的ngram
        raw_domain_words = list(d_ngram.keys())
        raw_domain_words.sort(key=lambda s: len(s), reverse=True)  # sort by len(str).
        domain_words_to_remove = []
        for i in range(0, len(d_ngram)):
            for j in range(i+1, len(d_ngram)):
                if raw_domain_words[i] in raw_domain_words[j]:
                    domain_words_to_remove.append(raw_domain_words[i])
                if raw_domain_words[j] in raw_domain_words[i]:
                    domain_words_to_remove.append(raw_domain_words[j])
        for r in domain_words_to_remove:
            try:
                del d_ngram[r]
            except:
                pass

        ''' 3 '''
        d_id = 0    #d_ngram 不是stopwords/septoken的stemmed_ngram:对应的起始位置列表
        for stemmed_ngram, start_end_list in d_ngram.items(): 
            d_id += 1
            for start, end in start_end_list:
                domain_ids[start:end] = [d_id] * (end - start)
                rebuilt_orin_ngram = " ".join(tokens[start: end])
                if not stemmed_ngram in domain_words_stemmed:
                    domain_words_stemmed[stemmed_ngram] = []
                if not rebuilt_orin_ngram in domain_words_orin:
                    domain_words_orin[rebuilt_orin_ngram] = []
                domain_words_stemmed[stemmed_ngram] +=  [(start, end)]
                domain_words_orin[rebuilt_orin_ngram] += [(start, end)]


        return domain_words_stemmed, domain_words_orin, domain_ids



    ''' start '''
    bpe_tokens_a = tokenizer.tokenize(text_a)  #tokens of context
    bpe_tokens_b = tokenizer.tokenize(text_b)  #tokens of question/one option
    bpe_tokens = [tokenizer.bos_token] + bpe_tokens_a + [tokenizer.sep_token] + \
                    bpe_tokens_b + [tokenizer.eos_token]  #tokens used for training

    #a_mask长度为max_length，+2是为了bos_token/eos_token也设置为1，a_mask除了context都mask掉
    a_mask = [1] * (len(bpe_tokens_a) + 2) + [0] * (max_length - (len(bpe_tokens_a) + 2))
    #b_mask长度为max_length，mask掉了context
    b_mask = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1) + [0] * (max_length - len(bpe_tokens))
    a_mask = a_mask[:max_length]
    b_mask = b_mask[:max_length]
    assert len(a_mask) == max_length, 'len_a_mask={}, max_len={}'.format(len(a_mask), max_length)
    assert len(b_mask) == max_length, 'len_b_mask={}, max_len={}'.format(len(b_mask), max_length)

    # adapting Ġ.
    assert isinstance(bpe_tokens, list)
    bare_tokens = [token[1:] if "Ġ" in token else token for token in bpe_tokens] #消除token开始符号"Ġ"
    argument_words, argument_space_ids = _find_arg_ngrams(bare_tokens, max_gram=max_gram)  
    domain_words_stemmed, domain_words_orin, domain_space_ids = _find_dom_ngrams_2(bare_tokens, max_gram=max_gram)  
    punct_space_ids = _find_punct(bare_tokens, punctuations)  

    argument_bpe_ids = argument_space_ids
    domain_bpe_ids = domain_space_ids
    punct_bpe_ids = punct_space_ids

    ''' output items '''
    input_ids = tokenizer.convert_tokens_to_ids(bpe_tokens) #模型的输入是token的id
    input_mask = [1] * len(input_ids) #1代表context+question+option
    segment_ids = [0] * (len(bpe_tokens_a) + 2) + [1] * (len(bpe_tokens_b) + 1)  #TODO:input_mask/segment_ids作用

    padding = [0] * (max_length - len(input_ids))
    padding_ids = [tokenizer.pad_token_id] * (max_length - len(input_ids))  #用填充符的id占位
    arg_dom_padding_ids = [-1] * (max_length - len(input_ids))  #
    input_ids += padding_ids
    argument_bpe_ids += arg_dom_padding_ids
    domain_bpe_ids += arg_dom_padding_ids
    punct_bpe_ids += arg_dom_padding_ids
    input_mask += padding
    segment_ids += padding

    input_ids = input_ids[:max_length]
    input_mask = input_mask[:max_length]
    segment_ids = segment_ids[:max_length]
    argument_bpe_ids = argument_bpe_ids[:max_length]
    domain_bpe_ids = domain_bpe_ids[:max_length]
    punct_bpe_ids = punct_bpe_ids[:max_length]

    assert len(input_ids) <= max_length, 'len_input_ids={}, max_length={}'.format(len(input_ids), max_length)
    assert len(input_mask) <= max_length, 'len_input_mask={}, max_length={}'.format(len(input_mask), max_length)
    assert len(segment_ids) <= max_length, 'len_segment_ids={}, max_length={}'.format(len(segment_ids), max_length)
    assert len(argument_bpe_ids) <= max_length, 'len_argument_bpe_ids={}, max_length={}'.format(
        len(argument_bpe_ids), max_length)
    assert len(domain_bpe_ids) <= max_length, 'len_domain_bpe_ids={}, max_length={}'.format(
        len(domain_bpe_ids), max_length)
    assert len(punct_bpe_ids) <= max_length, 'len_punct_bpe_ids={}, max_length={}'.format(
        len(punct_bpe_ids), max_length)

    output = {}
    output["input_tokens"] = bpe_tokens   #tokens used for training
    output["input_ids"] = input_ids   #模型的输入是token的id
    output["attention_mask"] = input_mask
    output["token_type_ids"] = segment_ids   #0代表context，1代表question+option
    output["argument_bpe_ids"] = argument_bpe_ids
    output["domain_bpe_ids"] = domain_bpe_ids
    output["punct_bpe_ids"] = punct_bpe_ids
    output["a_mask"] = a_mask
    output["b_mask"] = b_mask

    return output


def main(text, option):
    parser = argparse.ArgumentParser()
    parser.add_argument( 
        '--model_dir',
        type=str,
        # default='/data2/yinyahuang/BERT_MODELS/roberta-large-uncased'
        default='roberta-large'
        )
    parser.add_argument(
        '--max_gram',
        type=int,
        default=5,
        help="max ngram for window sliding."
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=256
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    stopwords = list(gensim.parsing.preprocessing.STOPWORDS) + punctuations

    inputs = arg_tokenizer(text, option, tokenizer, stopwords, relations, punctuations,
                           args.max_gram, args.max_length)


    ''' print '''
    p = []
    for token, arg, dom, pun in zip(inputs["input_tokens"], inputs["argument_bpe_ids"], inputs["domain_bpe_ids"],
                                    inputs["punct_bpe_ids"]):
        p.append((token, arg, dom, pun))  #每个token，是否是argument word，是否是domain word，是否是punc
    print(p)
    print('input_tokens（输入的所有token）\n{}'.format(inputs["input_tokens"]))
    print('input_ids（输入的所有token的id）\n{}, size={}'.format(inputs["input_ids"], len(inputs["input_ids"])))
    print('attention_mask（1代表context+question+option）\n{}'.format(inputs["attention_mask"]))
    print('token_type_ids（0代表context，1代表question+option）\n{}'.format(inputs["token_type_ids"]))
    print('argument_bpe_ids（0代表不是argument word,其它代表argument word对应的全局的pattern）\n{}'.format(inputs["argument_bpe_ids"]))
    print('domain_bpe_ids（0代表不是domain word，其它代表domain word对应的从1开始计数的非全局的id）\n{}, size={}'.format(inputs["domain_bpe_ids"], len(inputs["domain_bpe_ids"])))
    print('punct_bpe_ids（1代表是标点符号）\n{}'.format(inputs["punct_bpe_ids"]))

    import torch
    flat_punct_bpe_ids=torch.tensor(inputs["punct_bpe_ids"]).long()
    flat_argument_bpe_ids=torch.tensor(inputs["argument_bpe_ids"]).long()
    new_punct_id = 3 + 1
    new_punct_bpe_ids = new_punct_id * flat_punct_bpe_ids  # punct_id: 1 -> 4. for incorporating with argument_bpe_ids.
    _flat_all_bpe_ids = flat_argument_bpe_ids + new_punct_bpe_ids  # -1:padding, 0:non, 1-3: arg, 4:punct.
    overlapped_punct_argument_mask = (_flat_all_bpe_ids > new_punct_id).long()  #TODO:这什么意思，不应该是全false？
    flat_all_bpe_ids = _flat_all_bpe_ids * (1 - overlapped_punct_argument_mask) + flat_argument_bpe_ids * overlapped_punct_argument_mask
    assert flat_argument_bpe_ids.max().item() <= new_punct_id
    print(overlapped_punct_argument_mask)
    print(flat_all_bpe_ids)

if __name__ == '__main__':

    import json
    from graph_building_blocks.argument_set_punctuation_v4 import punctuations
    with open('./graph_building_blocks/explicit_arg_set_v4.json', 'r') as f:
        relations = json.load(f)  # key: relations, value: ignore

    text = 'Until he was dismissed amid great controversy, Hastings was considered one of the greatest intelligence ' \
           'agents of all time. It is clear that if his dismissal was justified, then Hastings was either ' \
           'incompetent or else disloyal. Soon after the dismissal, however, it was shown that he had never been ' \
           'incompetent. Thus, one is forced to conclude that Hastings must have been disloyal.'

    option = 'Everyone with an office on the second floor works directly for the president and, as a result, no one ' \
             'with a second floor office will take a July vacation because no one who works for the president will ' \
             'be able to take time off during July.'


    main(text, option)