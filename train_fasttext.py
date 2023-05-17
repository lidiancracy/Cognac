from gensim.models import FastText
from gensim.test.utils import common_texts
import json
import os
import pickle
from typing import List


def is_alpha(string):
    for x in string:
        if 64 < ord(x) and ord(x) < 123:
            continue
        else:
            return False
    return True

import json

def json2corpus(file_path):
    res = []
    with open(file_path, 'r', encoding='utf8') as f:
        for method_json in f.readlines():
            try:
                # 解析JSON数据
                method_data = json.loads(method_json)
                # 获取方法体中的所有元素，除了最后一个元素
                tokens = [token[0] for token in method_data[:-1]]
                # 获取最后一个元素作为方法名
                method_name = method_data[-1]
                # 创建当前行
                cur_line = tokens + method_name

                # 将当前行添加到结果列表
                res.append(cur_line)
                # print(cur_line)
            except:
                print("error")
                continue
    return res




def save_vocab_weight(model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'vocab.pkl'), 'wb') as f_vocab, open(os.path.join(output_dir, 'weight.pkl'), 'wb') as f_weight:
        pickle.dump(model.wv.index_to_key, f_vocab)
        pickle.dump(model.wv[model.wv.index_to_key], f_weight)


def main(corpus_path: List[str], output_dir):
    extra_words = [['<PAD>' for _ in range(min_count)] + ['<START>' for _ in range(min_count)] + \
                   ['<EOS>' for _ in range(min_count)] + ['<UNK>' for _ in range(min_count)]]
    model = FastText(vector_size=v_dim, window=window, min_count=min_count)  # 实例化
    model.build_vocab(corpus_iterable=extra_words)
    corpus = []
    for cp in corpus_path:
        corpus += json2corpus(cp)

    model.build_vocab(corpus_iterable=corpus, update=True)
    model.train(corpus_iterable=corpus, total_examples=len(corpus), epochs=10)
    save_vocab_weight(model, output_dir)

if __name__ == '__main__':
    v_dim = 128
    window = 8
    min_count = 1

    main(['.\\med_test\\train.json',
          '.\\med_test\\valid.json',
          '.\\med_test\\test.json'], './dataset/fasttext_vectors')
