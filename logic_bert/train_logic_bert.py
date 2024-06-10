import torch
torch.cuda.empty_cache()#oliver is to be blamed for this
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import os
import os.path
import json
import random
import argparse
from tqdm import tqdm

from model import LogicBERT

device = 'cpu'
RULES_THRESHOLD = 30

POSITION_EMB_FILE = 'position_emb.pt'
WORD_EMB_FILE = 'word_emb.pt'
MODEL_OUTPUT_FILE = 'model'
TRAIN_LOSS_LOG_FILE = 'train_loss_log.txt'
TEST_LOSS_LOG_FILE = 'test_loss_log.txt'
OTHER_DIST_LOSS_LOG_FILE = 'other_dist_loss_log.txt'
TRAIN_ACC_LOG_FILE = 'train_acc_log.txt'
TEST_ACC_LOG_FILE = 'test_acc_log.txt'
OTHER_DIST_ACC_LOG_FILE = 'other_dist_acc_log.txt'
PER_EPOCH_TRAIN_ACC_LOG_FILE = 'per_epoch_train_acc_log.txt'
PER_EPOCH_TEST_ACC_LOG_FILE = 'per_epoch_test_acc_log.txt'
PER_EPOCH_OTHER_DIST_ACC_LOG_FILE = 'per_epoch_other_dist_acc_log.txt'

class LogicDataset(Dataset):
    def __init__(self, examples):
        # self.examples = examples

        # skip examples that have too many rules
        self.examples = [ex for ex in examples if len(ex["rules"]) <= RULES_THRESHOLD]
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
    def initialze_from_file(cls, file, max_reasoning_depth):
        if "," in file:
            files = file.split(",")
        else:
            files = [file]
        all_examples = []
        for file_name in files:
            with open(file_name) as f:
                examples = json.load(f)
                for example in examples:
                    if example['depth'] <= max_reasoning_depth: # 3 -> 5 -> 4 (54oli) -> 5 (may13)
                        all_examples.extend([example])
            # with open(file_name) as f:
            #     examples = json.load(f)
            #     all_examples.extend(examples)
        return cls(all_examples)


def init():
    global device

    parser = argparse.ArgumentParser()

    
    parser.add_argument('--vocab_file', type=str, default='vocab.txt')
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--cuda_core', default='0', type=str)
    
    # parser.add_argument('--dataset_path', default='', type=str)
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--other_dist_data_file', default='', type=str) 
    # parser.add_argument('--dataset', default='', type=str) 
    parser.add_argument('--max_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--effective_batch_size', default=256, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=1.0, type=float)
    parser.add_argument('--optimizer', default='SGD', type=str)
    # parser.add_argument('--max_cluster_size', default=10, type=int)
    # parser.add_argument('--train_loss_log_file', default='train_loss_log.txt', type=str)
    # parser.add_argument('--test_loss_log_file', default='test_loss_log.txt', type=str)
    # parser.add_argument('--other_dist_loss_log_file', default='other_dist_loss_log.txt', type=str)
    # parser.add_argument('--train_acc_log_file', default='train_acc_log.txt', type=str)
    # parser.add_argument('--test_acc_log_file', default='test_acc_log.txt', type=str)
    # parser.add_argument('--other_dist_acc_log_file', default='other_dist_acc_log.txt', type=str)
    # parser.add_argument('--output_model_file', default='model.pt', type=str)
    parser.add_argument('--max_reasoning_depth', default=6, type=int)
    parser.add_argument('--model_layers', default=8, type=int)
    parser.add_argument('--experiment_directory', default='./', type=str)

    # parser.add_argument('--word_emb_file', type=str)
    # parser.add_argument('--position_emb_file', type=str)
    
    args = parser.parse_args()

    device = args.device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_core

    return args


# def read_vocab(vocab_file):
#     vocab = []
#     with open(vocab_file, 'r') as fin:
#         vocab = [line.strip() for line in fin.readlines()]
#     vocab += ['[CLS]', 'Start', 'Query', ':', 'Alice', 'is', '.', 'and', ',']
#     vocab = set(vocab)
#     print('vocabulary size: ', len(vocab))
#     return vocab


# def gen_word_embedding(vocab, word_emb_file):
#     word_emb = {}
#     for word in vocab:
#         word_emb[word] = torch.cat((torch.randn(59).to(device), torch.zeros(5).to(device)))
#         word_emb[word] /= torch.norm(word_emb[word])
#     word_emb['.'] = torch.cat((torch.zeros(59).to(device), torch.ones(1).to(device), torch.zeros(4).to(device)))
#     word_emb['[CLS]'] = word_emb['.']

#     torch.save(word_emb, word_emb_file)

#     return word_emb


# def position_embedding(n, position_emb_file):
#     P = torch.randn(n, 64).to(device)
#     for i in range(0, n):
#         P[i] /= torch.norm(P[i])
#     position_emb = torch.zeros(n, 768).to(device)
#     for i in range(1, n):
#         for j, k in enumerate([3, 5, 7, 1, 0, 2, 4, 6]):
#             if i - k >= 0:
#                 position_emb[i, 64*j:64*(j+1)] = P[i-k]

#     for j, k in enumerate([6, 4, 4, 6, 0, 4, 7, 7]):
#         position_emb[0, 64*j:64*(j+1)] = P[k]
    
#     torch.save(position_emb, position_emb_file)

#     return position_emb


# def tokenize_and_embed(sentence, word_emb, position_emb):
#     seq = [token for token in sentence.split(' ') if token != '']
#     x = torch.zeros(len(seq), 768).to(device)
#     for i, word in enumerate(seq):
#         x[i, :] = torch.cat((torch.zeros(64 * 8).to(device), word_emb[word], torch.zeros(64 * 3).to(device))) + position_emb[i]
#     return x
    

def train_model(model, optimizer, train, valid, test, other_dist,
                lr, weight_decay, batch_size, effective_batch_size, max_epoch,
                train_loss_log_file, test_loss_log_file, other_dist_loss_log_file, 
                train_acc_log_file, test_acc_log_file, other_dist_acc_log_file,
                per_epoch_train_acc_log_file, per_epoch_test_acc_log_file, per_epoch_other_dist_acc_log_file,
                output_model_file, max_reasoning_depth=6):
    valid_loader, test_loader = None, None
    train_loader = DataLoader(dataset=train, batch_size=batch_size, shuffle=True)
    if valid is not None:
        valid_loader = DataLoader(dataset=valid, batch_size=batch_size, shuffle=True)
    if test is not None:
        test_loader = DataLoader(dataset=test, batch_size=batch_size, shuffle=True)
    if other_dist is not None:
        other_dist_loader = DataLoader(dataset=other_dist, batch_size=batch_size, shuffle=True)

    # select optimizer and loss function
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    bce_loss = torch.nn.BCELoss()                    # use Binary Cross Entropy Loss 

    # set up logging files
    init_log_files(train_loss_log_file, test_loss_log_file, other_dist_loss_log_file, 
                train_acc_log_file, test_acc_log_file, other_dist_acc_log_file,
                per_epoch_train_acc_log_file, per_epoch_test_acc_log_file, per_epoch_other_dist_acc_log_file)

    # training loop
    model = model.to(device)
    model.train()
    for epoch in tqdm(range(0, max_epoch)):
        # compute the model accuracy each epoch:
        train_acc, train_acc_by_depth = evaluate_by_depth(model, train_loader, max_reasoning_depth)
        test_acc, test_acc_by_depth = evaluate_by_depth(model, test_loader, max_reasoning_depth)
        other_dist_acc, other_dist_acc_by_depth = evaluate_by_depth(model, other_dist_loader, max_reasoning_depth)

        # # print the accuracies
        # print('Epoch {}; train acc: {}; test acc: {}, other dist acc: {}'.format(epoch, train_acc, test_acc, other_dist_acc))

        # output the accuracies to the log files
        accs = [train_acc, test_acc, other_dist_acc]
        accs_by_depth = [train_acc_by_depth, test_acc_by_depth, other_dist_acc_by_depth]
        filenames = [per_epoch_train_acc_log_file, per_epoch_test_acc_log_file, per_epoch_other_dist_acc_log_file]
        for acc,acc_by_depth,filename in zip(accs,accs_by_depth,filenames):
            with open(filename, 'a+') as f:
                s = "{} {} "
                for i in range(max_reasoning_depth+1):
                    s = s + '{} '
                s = s + "\n"
                f.write(s.format(epoch, acc, *acc_by_depth))

        # save the model entering each epoch (checkpoint)
        if output_model_file != '':
            torch.save(model, output_model_file + '_epoch' + str(epoch) + '.pt')

        # training loop
        accum_iter = effective_batch_size
        # accumulate loss (by depth)
        batch_loss_by_depth = [0 for i in range(max_reasoning_depth+1)]
        # accumulate accuracy (by depth)
        correct_count_by_depth = [0 for i in range(max_reasoning_depth+1)]
        total_count_by_depth = [0 for i in range(max_reasoning_depth+1)]
        for batch_idx, (x_batch, labels, ex_depth) in enumerate(train_loader):#enumerate(tqdm(train_loader))
            # forward passes
            y_batch = []
            for sentence,label,ex_depth in zip(x_batch,labels,ex_depth):# since the realized batch size during gradient accumulation is 1, this loop is over 1 item only
                # input_state = tokenize_and_embed(sentence, word_emb, position_emb)
                m_out = model(sentence)
                y = m_out[0, 255]
                correct_prediction = ((y>.5) == label)
                correct_count_by_depth[ex_depth] += correct_prediction
                total_count_by_depth[ex_depth] += 1
                # for computing the loss we also track the prediction
                y_batch.append(y)
            
            # compute loss
            y_batch = torch.stack(y_batch, dim=0)
            y_batch = y_batch.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)     
            loss = bce_loss(y_batch, labels) / accum_iter                   # normalize by the number of iterations of accumulation
            loss.backward()                                                 # compute gradients (this adds to previously accumulated gradients)
            batch_loss_by_depth[ex_depth] += loss.item() * accum_iter       # track loss by depth for logging (normalized by depth count later)

            # print the loss for this batch
            # print('TRAIN: depth',ex_depth[0].item(),', label',labels.item(), ', predict',y_batch.item(),', correct', correct_prediction.item(), ', loss',loss.item())

            if ((batch_idx + 1) % accum_iter == 0): #or (batch_idx + 1 == len(train_loader)):
                # compute accumulated losses for this batch; check loss/accuracy on other datasets; finally, an do SGD step
                batch_loss = sum(batch_loss_by_depth) / sum(total_count_by_depth)
                accumulated_loss_by_depth = []
                for i in range(len(correct_count_by_depth)):
                    if total_count_by_depth[i] > 0:
                        accumulated_loss_by_depth.append(batch_loss_by_depth[i] / total_count_by_depth[i])
                    else:
                        accumulated_loss_by_depth.append(0)

                # log the losses (by depth)
                with open(train_loss_log_file, 'a+') as f:
                    s = "{} {} "
                    for i in range(len(accumulated_loss_by_depth)):
                        s = s + '{} '
                    s = s + "\n"
                    f.write(s.format(epoch, batch_loss, *accumulated_loss_by_depth))#* should make the elements of that list interepreted as separated arguments and therefore output as space-separated like tracy's script wants
                
                # compute accumulated accuracies for this batch
                acc = sum(correct_count_by_depth) / sum(total_count_by_depth)
                acc_by_depth = []
                for i in range(len(correct_count_by_depth)):
                    if total_count_by_depth[i] > 0:
                        acc_by_depth.append(correct_count_by_depth[i] / total_count_by_depth[i])
                    else:
                        acc_by_depth.append(0)
                
                # log the accuracies (by depth)
                with open(train_acc_log_file, 'a+') as f:
                    s = "{} {} "
                    for i in range(len(acc_by_depth)):
                        s = s + '{} '
                    s = s + "\n"
                    f.write(s.format(epoch, acc, *acc_by_depth))

                # print an update on the progress
                print('Train, Epoch {}, Batch loss {}, Batch accuracy {}'.format(epoch,batch_loss,acc))

                # reset the batch losses and accuracies
                batch_loss_by_depth = [0 for i in range(max_reasoning_depth+1)]
                correct_count_by_depth = [0 for i in range(max_reasoning_depth+1)]
                total_count_by_depth = [0 for i in range(max_reasoning_depth+1)]

                # before continuing to the next effective batch,
                # also compute the loss on the test set
                batch_idx_inner = 0
                while True:
                    batch_idx_inner += 1
                    (x_batch, labels, ex_depth) = next(iter(test_loader))
                    # forward passes
                    y_batch = []
                    for sentence,label,ex_depth in zip(x_batch,labels,ex_depth):# since the realized batch size during gradient accumulation is 1, this loop is over 1 item only
                        # input_state = tokenize_and_embed(sentence, word_emb, position_emb)
                        m_out = model(sentence)
                        y = m_out[0, 255]
                        correct_prediction = ((y>.5) == label)
                        correct_count_by_depth[ex_depth] += correct_prediction
                        total_count_by_depth[ex_depth] += 1
                        # for computing the loss we also track the prediction
                        y_batch.append(y)
                    
                    # compute loss
                    y_batch = torch.stack(y_batch, dim=0)
                    y_batch = y_batch.type(torch.FloatTensor)
                    labels = labels.type(torch.FloatTensor)     
                    loss = bce_loss(y_batch, labels) / accum_iter                   # normalize by the number of iterations of accumulation
                    batch_loss_by_depth[ex_depth] += loss.item() * accum_iter       # track loss by depth for logging (normalized by depth count later)

                    # print('TEST: depth',ex_depth[0].item(),', label',labels.item(), ', predict',y_batch.item(), ', loss',loss.item())

                    if ((batch_idx_inner + 1) % accum_iter == 0): #or (batch_idx + 1 == len(train_loader)):
                        # compute accumulated losses for this batch
                        batch_loss = sum(batch_loss_by_depth) / sum(total_count_by_depth)
                        accumulated_loss_by_depth = []
                        for i in range(len(correct_count_by_depth)):
                            if total_count_by_depth[i] > 0:
                                accumulated_loss_by_depth.append(batch_loss_by_depth[i] / total_count_by_depth[i])
                            else:
                                accumulated_loss_by_depth.append(0)

                        # log losses (by depth)
                        with open(test_loss_log_file, 'a+') as f:
                            s = "{} {} "
                            for i in range(len(accumulated_loss_by_depth)):
                                s = s + '{} '
                            s = s + "\n"
                            f.write(s.format(epoch, batch_loss, *accumulated_loss_by_depth))

                        # compute accumulated accuracies for this batch
                        acc = sum(correct_count_by_depth) / sum(total_count_by_depth)
                        acc_by_depth = []
                        for i in range(len(correct_count_by_depth)):
                            if total_count_by_depth[i] > 0:
                                acc_by_depth.append(correct_count_by_depth[i] / total_count_by_depth[i])
                            else:
                                acc_by_depth.append(0)
                
                        # log accuracies (by depth)
                        with open(test_acc_log_file, 'a+') as f:
                            s = "{} {} "
                            for i in range(len(acc_by_depth)):
                                s = s + '{} '
                            s = s + "\n"
                            f.write(s.format(epoch, acc, *acc_by_depth))  
                        
                        # print an update on the progress
                        print('Test, Epoch {}, Batch loss {}, Batch accuracy {}'.format(epoch,batch_loss,acc))
                        
                        # reset the batch losses and accuracies
                        batch_loss_by_depth = [0 for i in range(max_reasoning_depth+1)]
                        correct_count_by_depth = [0 for i in range(max_reasoning_depth+1)]
                        total_count_by_depth = [0 for i in range(max_reasoning_depth+1)]
                        break

                # now repeat (compute loss) on the other_dist dataset
                batch_idx_inner = 0
                while True:
                    batch_idx_inner += 1
                    (x_batch, labels, ex_depth) = next(iter(other_dist_loader))
                    # forward passes
                    y_batch = []
                    for sentence,label,ex_depth in zip(x_batch,labels,ex_depth):# since the realized batch size during gradient accumulation is 1, this loop is over 1 item only
                        # input_state = tokenize_and_embed(sentence, word_emb, position_emb)
                        m_out = model(sentence)
                        y = m_out[0, 255]
                        correct_prediction = ((y>.5) == label)
                        correct_count_by_depth[ex_depth] += correct_prediction
                        total_count_by_depth[ex_depth] += 1
                        # for computing the loss we also track the prediction
                        y_batch.append(y)
                    
                    # compute loss
                    y_batch = torch.stack(y_batch, dim=0)
                    y_batch = y_batch.type(torch.FloatTensor)
                    labels = labels.type(torch.FloatTensor)     
                    loss = bce_loss(y_batch, labels) / accum_iter                   # normalize by the number of iterations of accumulation
                    batch_loss_by_depth[ex_depth] += loss.item() * accum_iter       # track loss by depth for logging (normalized by depth count later)

                    # print('OTHER DIST: depth',ex_depth[0].item(),', label',labels.item(), ', predict',y_batch.item(), ', loss',loss.item())

                    if ((batch_idx_inner + 1) % accum_iter == 0): #or (batch_idx + 1 == len(train_loader)):
                        # compute accumulated losses for this batch
                        batch_loss = sum(batch_loss_by_depth) / sum(total_count_by_depth)
                        accumulated_loss_by_depth = []
                        for i in range(len(correct_count_by_depth)):
                            if total_count_by_depth[i] > 0:
                                accumulated_loss_by_depth.append(batch_loss_by_depth[i] / total_count_by_depth[i])
                            else:
                                accumulated_loss_by_depth.append(0)

                        # log losses (by depth)
                        with open(other_dist_loss_log_file, 'a+') as f:
                            s = "{} {} "
                            for i in range(len(accumulated_loss_by_depth)):
                                s = s + '{} '
                            s = s + "\n"
                            f.write(s.format(epoch, batch_loss, *accumulated_loss_by_depth))

                        # compute accumulated accuracies for this batch
                        acc = sum(correct_count_by_depth) / sum(total_count_by_depth)
                        acc_by_depth = []
                        for i in range(len(correct_count_by_depth)):
                            if total_count_by_depth[i] > 0:
                                acc_by_depth.append(correct_count_by_depth[i] / total_count_by_depth[i])
                            else:
                                acc_by_depth.append(0)
                        
                        # log accuracies (by depth)
                        with open(other_dist_acc_log_file, 'a+') as f:
                            s = "{} {} "
                            for i in range(len(acc_by_depth)):
                                s = s + '{} '
                            s = s + "\n"
                            f.write(s.format(epoch, acc, *acc_by_depth))  
                        
                        # print an update on the progress
                        print('Other distribution, Epoch {}, Batch loss {}, Batch accuracy {}'.format(epoch,batch_loss,acc))
                        
                        # reset the batch losses and accuracies
                        batch_loss_by_depth = [0 for i in range(max_reasoning_depth+1)]
                        correct_count_by_depth = [0 for i in range(max_reasoning_depth+1)]
                        total_count_by_depth = [0 for i in range(max_reasoning_depth+1)]
                        break

                # finally, do an SGD update based on the accumulated gradient (only from the training data)
                optimizer.step()
                # and now zero out the gradient to begin accumulating again 
                optimizer.zero_grad()

        # finally, save the model each epoch
        if output_model_file != '':
            torch.save(model, output_model_file + '_epoch' + str(epoch) + '.pt')

    
    # do the final epoch checkpoint (compute accuracy and save the model)
    epoch = max_epoch

    # compute the model accuracy each epoch:
    train_acc, train_acc_by_depth = evaluate_by_depth(model, train_loader, max_reasoning_depth)
    test_acc, test_acc_by_depth = evaluate_by_depth(model, test_loader, max_reasoning_depth)
    other_dist_acc, other_dist_acc_by_depth = evaluate_by_depth(model, other_dist_loader, max_reasoning_depth)

    # # print the accuracies
    # print('Epoch {}; train acc: {}; test acc: {}, other dist acc: {}'.format(epoch, train_acc, test_acc, other_dist_acc))

    # output the accuracies to the log files
    accs = [train_acc, test_acc, other_dist_acc]
    accs_by_depth = [train_acc_by_depth, test_acc_by_depth, other_dist_acc_by_depth]
    filenames = [per_epoch_train_acc_log_file, per_epoch_test_acc_log_file, per_epoch_other_dist_acc_log_file]
    for acc,acc_by_depth,filename in zip(accs,accs_by_depth,filenames):
        with open(filename, 'a+') as f:
            s = "{} {} "
            for i in range(max_reasoning_depth+1):
                s = s + '{} '
            s = s + "\n"
            f.write(s.format(epoch, acc, *acc_by_depth))

    # save the model entering each epoch (checkpoint)
    if output_model_file != '':
        torch.save(model, output_model_file + '_epoch' + str(epoch) + '.pt')


# def evaluate(model, dataset_loader, word_emb, position_emb):
#     accs = []
#     dataset_len = 0
#     counter = 0
#     for x_batch, labels, ex_depth in dataset_loader:
#         batch_correct = 0
#         batch_total = 0
#         for sentence,label in zip(x_batch,labels):
#             input_state = tokenize_and_embed(sentence, word_emb, position_emb)
#             m_out = model(input_state)
#             y = m_out[0, 255]
#             correct_prediction = ((y>.5) == label)
#             accs.append(correct_prediction)
#             batch_correct += correct_prediction
#             batch_total += 1
#     print("sum:", sum(accs))
#     acc = sum(accs).item() / len(accs)
#     return acc

def init_log_files(train_loss_log_file, test_loss_log_file, other_dist_loss_log_file, 
                train_acc_log_file, test_acc_log_file, other_dist_acc_log_file,
                per_epoch_train_acc_log_file, per_epoch_test_acc_log_file, per_epoch_other_dist_acc_log_file):
    # (first check if this will overwrite existing log files)
    if os.path.isfile(train_loss_log_file):
        input("You are about to overwrite existing log files; press any key to continue")
    with open(train_loss_log_file, 'w') as f:
        f.write('Epoch | Accum Batch Loss | Batch Loss by Depth ...\n')
    with open(test_loss_log_file, 'w') as f:
        f.write('Epoch | Accum Batch Loss | Batch Loss by Depth ...\n')
    with open(other_dist_loss_log_file, 'w') as f:
        f.write('Epoch | Accum Batch Loss | Batch Loss by Depth ...\n')
    with open(train_acc_log_file, 'w') as f:
        f.write('Epoch | Train Acc | Batch Acc by Depth ...\n')
    with open(test_acc_log_file, 'w') as f:
        f.write('Epoch | Test Acc | Batch Acc by Depth ...\n')
    with open(other_dist_acc_log_file, 'w') as f:
        f.write('Epoch | Other Dist Acc | Batch Acc by Depth ...\n')
    with open(per_epoch_train_acc_log_file, 'w') as f:
        f.write('Epoch | Per Epoch Acc | Per Epoch Acc by Depth ...\n')
    with open(per_epoch_test_acc_log_file, 'w') as f:
        f.write('Epoch | Per Epoch Acc | Per Epoch Acc by Depth ...\n')
    with open(per_epoch_other_dist_acc_log_file, 'w') as f:
        f.write('Epoch | Per Epoch Acc | Per Epoch Acc by Depth ...\n')

def evaluate_by_depth(model, dataset_loader, max_reasoning_depth):
    # accumulate accuracy (by depth)
    correct_count_by_depth = [0 for i in range(max_reasoning_depth+1)]
    total_count_by_depth = [0 for i in range(max_reasoning_depth+1)]
    for x_batch, labels, ex_depths in dataset_loader:
        for sentence, label, ex_depth in zip(x_batch, labels, ex_depths):
            # input_state = tokenize_and_embed(sentence, word_emb, position_emb)
            m_out = model(sentence)
            y = m_out[0, 255]
            correct_prediction = ((y>.5) == label)
            correct_count_by_depth[ex_depth] += correct_prediction
            total_count_by_depth[ex_depth] += 1
    # compute accuracies
    acc = sum(correct_count_by_depth) / sum(total_count_by_depth)
    acc_by_depth = []
    for i in range(len(correct_count_by_depth)):
        if total_count_by_depth[i] > 0:
            acc_by_depth.append(correct_count_by_depth[i] / total_count_by_depth[i])
        else:
            acc_by_depth.append(0)
    return acc, acc_by_depth

def main():
    args = init()

    train = LogicDataset.initialze_from_file(args.data_file+'_train', args.max_reasoning_depth)
    valid = LogicDataset.initialze_from_file(args.data_file+'_val', args.max_reasoning_depth)
    test = LogicDataset.initialze_from_file(args.data_file+'_test', args.max_reasoning_depth)
    other_dist = LogicDataset.initialze_from_file(args.other_dist_data_file, args.max_reasoning_depth)
    
    # vocab = read_vocab(args.vocab_file)
    # word_emb = gen_word_embedding(vocab, args.experiment_directory + WORD_EMB_FILE)
    # position_emb = gen_position_embedding(1024, args.experiment_directory + POSITION_EMB_FILE)

    model = LogicBERT(vocab_file=args.vocab_file, model_layers=args.model_layers, device=device)
    model.to(device)
 
    # save a log file with this experiment's details
    details = {}
    details["experiment_directory"] = args.experiment_directory
    details["max_reasoning_depth"] = args.max_reasoning_depth
    details["model_layers"] = args.model_layers
    details["lr"] = args.lr
    details["batch_size"] = args.batch_size
    details["effective_batch_size"] = args.effective_batch_size
    details["data_file"] = args.data_file
    details["other_dist_data_file"] = args.other_dist_data_file
    details["max_epoch"] = args.max_epoch
    details["weight_decay"] = args.weight_decay
    details["optimizer"] = args.optimizer
    with open(args.experiment_directory + "experiment_details.json", "w") as file:
        json.dump(details, file)


    train_model(model, optimizer=args.optimizer, train=train, valid=valid, test=test, other_dist=other_dist,
        lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, effective_batch_size=args.effective_batch_size, max_epoch=args.max_epoch,
        train_loss_log_file=args.experiment_directory + TRAIN_LOSS_LOG_FILE, test_loss_log_file=args.experiment_directory + TEST_LOSS_LOG_FILE, other_dist_loss_log_file=args.experiment_directory + OTHER_DIST_LOSS_LOG_FILE, 
        train_acc_log_file=args.experiment_directory + TRAIN_ACC_LOG_FILE, test_acc_log_file=args.experiment_directory + TEST_ACC_LOG_FILE, other_dist_acc_log_file=args.experiment_directory + OTHER_DIST_ACC_LOG_FILE, 
        per_epoch_train_acc_log_file=args.experiment_directory + PER_EPOCH_TRAIN_ACC_LOG_FILE, 
        per_epoch_test_acc_log_file=args.experiment_directory + PER_EPOCH_TEST_ACC_LOG_FILE, 
        per_epoch_other_dist_acc_log_file=args.experiment_directory + PER_EPOCH_OTHER_DIST_ACC_LOG_FILE,         
        output_model_file=args.experiment_directory + MODEL_OUTPUT_FILE,
        max_reasoning_depth=args.max_reasoning_depth)

if __name__ == '__main__':
    main()
