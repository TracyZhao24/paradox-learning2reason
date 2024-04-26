import torch
torch.cuda.empty_cache()#oliver is to be blamed for this
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
import json
import random
import argparse
from tqdm import tqdm

from model import LogicBERT

device = 'cpu'
RULES_THRESHOLD = 30
# DEPTH = 5

class LogicDataset(Dataset):
    def __init__(self, examples, depth):
        # self.examples = examples
        # skip examples that have too many rules
        # self.examples = [ex for ex in examples if len(ex["rules"]) <= RULES_THRESHOLD]
        # filter by depth
        self.examples = [ex for ex in examples if len(ex["rules"]) <= RULES_THRESHOLD and (ex["depth"]) == depth]
        # print(len(self.examples))
        random.shuffle(self.examples)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        # [CLS] Query : Alice is C . A . C . A and B , C .
        text = ""
        text += "[CLS] Start Query : "
        text += "Alice is " + example["query"] + " . "

        for fact in example["facts"]:
            text +=  fact
            text += " . "

        for rule in example["rules"]:
            text +=  " and ".join(rule[0])
            text += " , "
            text += rule[-1]
            text += " . "

        return text, example["label"], example["depth"]


    @classmethod
    def initialze_from_file(cls, file, depth):
        if "," in file:
            files = file.split(",")
        else:
            files = [file]
        all_examples = []
        for file_name in files:
            with open(file_name) as f:
                examples = json.load(f)
                for example in examples:
                    if example['depth'] < 6:
                        all_examples.extend([example])
            # with open(file_name) as f:
            #     examples = json.load(f)
            #     all_examples.extend(examples)
        return cls(all_examples, depth)


def init():
    global device

    parser = argparse.ArgumentParser()

    # was already here
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--word_emb_path', type=str)
    parser.add_argument('--position_emb_path', type=str)

    parser.add_argument('--vocab_file', type=str, default='vocab.txt')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--cuda_core', default='0', type=str)
    
    # oliver added
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--max_cluster_size', default=10, type=int)
    parser.add_argument('--log_file', default='log.txt', type=str)

    # tracy added
    parser.add_argument('--depth', type=int)
    
    args = parser.parse_args()

    device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_core

    return args


def read_vocab(vocab_file):
    vocab = []
    with open(vocab_file, 'r') as fin:
        vocab = [line.strip() for line in fin.readlines()]
    vocab += ['[CLS]', 'Start', 'Query', ':', 'Alice', 'is', '.', 'and', ',']
    vocab = set(vocab)
    print('vocabulary size: ', len(vocab))
    return vocab


def gen_word_embedding(vocab):
    word_emb = {}
    for word in vocab:
        word_emb[word] = torch.cat((torch.randn(59).to(device), torch.zeros(5).to(device)))
        word_emb[word] /= torch.norm(word_emb[word])
    word_emb['.'] = torch.cat((torch.zeros(59).to(device), torch.ones(1).to(device), torch.zeros(4).to(device)))
    word_emb['[CLS]'] = word_emb['.']

    return word_emb


def gen_position_embedding(n):
    P = torch.randn(n, 64).to(device)
    for i in range(0, n):
        P[i] /= torch.norm(P[i])
    position_emb = torch.zeros(n, 768).to(device)
    for i in range(1, n):
        for j, k in enumerate([3, 5, 7, 1, 0, 2, 4, 6]):
            if i - k >= 0:
                position_emb[i, 64*j:64*(j+1)] = P[i-k]

    for j, k in enumerate([6, 4, 4, 6, 0, 4, 7, 7]):
        position_emb[0, 64*j:64*(j+1)] = P[k]

    return position_emb


def tokenize_and_embed(sentence, word_emb, position_emb):
    seq = [token for token in sentence.split(' ') if token != '']
    x = torch.zeros(len(seq), 768).to(device)
    for i, word in enumerate(seq):
        x[i, :] = torch.cat((torch.zeros(64 * 8).to(device), word_emb[word], torch.zeros(64 * 3).to(device))) + position_emb[i]
    return x
    

# based on PGC repo: pgc/train.py
def test_model(model, test, batch_size,
                log_file, word_emb, position_emb, datafile, depth):
    
    test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)

    # training loop
    model = model.to(device)
    # compute accuracy on train, valid and test
    with torch.no_grad():
        test_acc = evaluate(model, test_loader, word_emb, position_emb)

    print('test acc: {}'.format(test_acc))

    with open(log_file, 'a+') as f:
        f.write('{} rules threshold, depth {} for {}: {} \n'.format(RULES_THRESHOLD, depth, datafile, test_acc))


def evaluate(model, dataset_loader, word_emb, position_emb):
    accs = []
    dataset_len = 0
    counter = 0
    for x_batch, labels, something_else_lol in dataset_loader:
        # counter += 1
        # if counter > 10:
        #     break
        #x_batch = x_batch.to(device)
        #y_batch = []
        batch_correct = 0
        batch_total = 0
        for sentence,label in zip(x_batch,labels):
            input_state = tokenize_and_embed(sentence, word_emb, position_emb)
            m_out = model(input_state)
            y = m_out[0, 255]
            correct_prediction = ((y>.5) == label)
            accs.append(correct_prediction)
            batch_correct += correct_prediction
            batch_total += 1
        #print('evaluation batch accuracy: {}'.format(batch_correct/batch_total))
    acc = sum(accs).item() / len(accs)
    return acc

def main():
    args = init()

    # evaluate by example depth
    test = LogicDataset.initialze_from_file(args.data_file+'_test', args.depth)

    # load embeddings from training
    word_emb = torch.load(args.word_emb_path)
    position_emb = torch.load(args.position_emb_path)

    # evaluate LogicBERT on all depths
    # model = LogicBERT()
    model = torch.load(args.model_path)
    model.to(device)

    print(f'now testing depth {args.depth}')
    
    test_model(model, test=test,
        batch_size=args.batch_size,
        log_file=args.log_file, word_emb=word_emb,
        position_emb=position_emb, datafile = args.data_file+'_test', depth = args.depth)

    """ old code (evaluate.py) that checks the model's correctness
    correct_counter = 0

    for index in tqdm(range(len(val_dataset))):        
        text, label, depth = val_dataset[index]

        # skip examples of depth > 10
        if depth > 10:
            continue

        with torch.no_grad():
            input_states = tokenize_and_embed(text, word_emb, position_emb)
            output = model(input_states)

            if (output[0, 255].item() > 0.5) == label:
                correct_counter += 1                
            else:
                print('Wrong Answer!')
                exit(0)

    print(f'AC: {correct_counter} tests passed')
    """

if __name__ == '__main__':
    main()
