import os
import sys
import pickle
import pandas as pd
import numpy as np
import jieba
from tqdm import tqdm
from collections import Counter
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = Counter()
        
    def augment_neutral_data(self, df):
        neutral_df = df[df['label'] == 2].copy()
        if len(neutral_df) == 0:
            return df
        
        print(f"\n原始中性数据量: {len(neutral_df)}")
        
        augmented_data = []
        for idx, row in neutral_df.iterrows():
            text = row['review']
            tokens = self.tokenize_text(text)
            
            if len(tokens) < 3:
                continue
            
            augmented_texts = []
            
            random.shuffle(tokens)
            shuffled_text = ''.join(tokens)
            augmented_texts.append(shuffled_text)
            
            if len(tokens) >= 5:
                num_to_drop = max(1, len(tokens) // 10)
                for _ in range(2):
                    temp_tokens = tokens.copy()
                    drop_indices = random.sample(range(len(temp_tokens)), min(num_to_drop, len(temp_tokens)))
                    drop_indices.sort(reverse=True)
                    for idx_drop in drop_indices:
                        temp_tokens.pop(idx_drop)
                    if len(temp_tokens) >= 3:
                        augmented_texts.append(''.join(temp_tokens))
            
            for aug_text in augmented_texts[:3]:
                augmented_data.append({
                    'label': 2,
                    'review': aug_text
                })
        
        if augmented_data:
            aug_df = pd.DataFrame(augmented_data)
            df = pd.concat([df, aug_df], ignore_index=True)
            print(f"增强后中性数据量: {len(df[df['label'] == 2])}")
            print(f"新增数据量: {len(augmented_data)}")
        
        return df
        
    def load_data(self):
        data_files = [
            os.path.join(self.config.DATA_DIR, 'ChnSentiCorp_htl_all.csv'),
            os.path.join(self.config.DATA_DIR, 'waimai_10k.csv'),
            os.path.join(self.config.DATA_DIR, 'weibo_senti_100k.csv'),
            os.path.join(self.config.DATA_DIR, 'neutral.csv')
        ]
        
        all_data = []
        
        for file_path in data_files:
            if os.path.exists(file_path):
                print(f"正在加载: {file_path}")
                df = pd.read_csv(file_path)
                all_data.append(df)
                print(f"  加载了 {len(df)} 条数据")
            else:
                print(f"警告: 文件不存在 - {file_path}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n总共加载了 {len(combined_df)} 条数据")
            return combined_df
        else:
            raise ValueError("没有找到任何数据文件")
    
    def tokenize_text(self, text):
        if pd.isna(text):
            return []
        text = str(text).strip()
        if not text:
            return []
        tokens = list(jieba.cut(text))
        tokens = [token for token in tokens if token.strip()]
        return tokens
    
    def build_vocab(self, texts):
        print("\n正在构建词表...")
        
        for text in tqdm(texts, desc="分词中"):
            tokens = self.tokenize_text(text)
            self.word_freq.update(tokens)
        
        print(f"总词数: {len(self.word_freq)}")
        
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words[:self.config.VOCAB_SIZE - 2]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        print(f"词表大小: {len(self.word2idx)}")
        print(f"词表覆盖率: {sum(freq for word, freq in sorted_words[:self.config.VOCAB_SIZE - 2]) / sum(self.word_freq.values()) * 100:.2f}%")
    
    def text_to_sequence(self, text):
        tokens = self.tokenize_text(text)
        sequence = [self.word2idx.get(token, self.word2idx['<UNK>']) for token in tokens]
        
        if len(sequence) > self.config.MAX_LEN:
            sequence = sequence[:self.config.MAX_LEN]
        else:
            sequence = sequence + [self.word2idx['<PAD>']] * (self.config.MAX_LEN - len(sequence))
        
        return sequence
    
    def preprocess(self):
        print("=" * 50)
        print("开始数据预处理")
        print("=" * 50)
        
        df = self.load_data()
        
        print(f"\n数据集列名: {df.columns.tolist()}")
        print(f"标签分布:\n{df['label'].value_counts()}")
        
        df = df.dropna(subset=['review'])
        df = df[df['review'].str.strip() != '']
        
        print(f"\n清理后数据量: {len(df)}")
        
        df = self.augment_neutral_data(df)
        
        self.build_vocab(df['review'].tolist())
        
        print("\n正在将文本转换为序列...")
        sequences = []
        for text in tqdm(df['review'].tolist(), desc="序列化中"):
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
        
        df['sequence'] = sequences
        df['sequence_length'] = df['sequence'].apply(lambda x: len([s for s in x if s != 0]))
        
        print(f"\n序列长度统计:")
        print(f"  平均长度: {df['sequence_length'].mean():.2f}")
        print(f"  最大长度: {df['sequence_length'].max()}")
        print(f"  最小长度: {df['sequence_length'].min()}")
        print(f"  中位数长度: {df['sequence_length'].median():.2f}")
        
        return df
    
    def split_data(self, df):
        np.random.seed(self.config.RANDOM_SEED)
        df = df.sample(frac=1, random_state=self.config.RANDOM_SEED).reset_index(drop=True)
        
        total_size = len(df)
        train_size = int(total_size * self.config.TRAIN_RATIO)
        val_size = int(total_size * self.config.VAL_RATIO)
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        print(f"\n数据集划分:")
        print(f"  训练集: {len(train_df)} 条 ({len(train_df)/total_size*100:.1f}%)")
        print(f"  验证集: {len(val_df)} 条 ({len(val_df)/total_size*100:.1f}%)")
        print(f"  测试集: {len(test_df)} 条 ({len(test_df)/total_size*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def save_vocab(self):
        print(f"\n正在保存词表...")
        
        with open(self.config.WORD2IDX_FILE, 'wb') as f:
            pickle.dump(self.word2idx, f)
        print(f"  word2idx 保存到: {self.config.WORD2IDX_FILE}")
        
        with open(self.config.IDX2WORD_FILE, 'wb') as f:
            pickle.dump(self.idx2word, f)
        print(f"  idx2word 保存到: {self.config.IDX2WORD_FILE}")
    
    def save_data(self, train_df, val_df, test_df):
        print(f"\n正在保存预处理数据...")
        
        data = {
            'train': {
                'sequences': np.array(train_df['sequence'].tolist()),
                'labels': np.array(train_df['label'].tolist()),
                'texts': train_df['review'].tolist()
            },
            'val': {
                'sequences': np.array(val_df['sequence'].tolist()),
                'labels': np.array(val_df['label'].tolist()),
                'texts': val_df['review'].tolist()
            },
            'test': {
                'sequences': np.array(test_df['sequence'].tolist()),
                'labels': np.array(test_df['label'].tolist()),
                'texts': test_df['review'].tolist()
            }
        }
        
        with open(self.config.PREPROCESSED_DATA_FILE, 'wb') as f:
            pickle.dump(data, f)
        print(f"  预处理数据保存到: {self.config.PREPROCESSED_DATA_FILE}")
    
    def run(self):
        df = self.preprocess()
        train_df, val_df, test_df = self.split_data(df)
        self.save_vocab()
        self.save_data(train_df, val_df, test_df)
        
        print("\n" + "=" * 50)
        print("数据预处理完成!")
        print("=" * 50)
        
        return df, train_df, val_df, test_df


def main():
    print("检查GPU可用性...")
    import torch
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("GPU不可用，将使用CPU")
    
    print(f"\n使用设备: {Config.DEVICE}")
    
    preprocessor = DataPreprocessor(Config)
    df, train_df, val_df, test_df = preprocessor.run()
    
    print(f"\n最终数据集信息:")
    print(f"  总数据量: {len(df)}")
    print(f"  词表大小: {len(preprocessor.word2idx)}")
    print(f"  最大序列长度: {Config.MAX_LEN}")
    print(f"  分类数: {Config.NUM_CLASSES}")


if __name__ == '__main__':
    main()
