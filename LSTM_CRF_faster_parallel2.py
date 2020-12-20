# author:时间：2020/7/8

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


def to_scalar(var):  # var是Variable,维度是１
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return idxs
    # tensor = torch.LongTensor(idxs)
    # return autograd.Variable(tensor)


def padding(instance_data, instance_label):
    len_max = 1
    for instance_id in range(len(instance_data)):
        if len(instance_data[instance_id]) > len_max:
            len_max = len(instance_data[instance_id])
    # print(len_max)
    if len(instance_data) == 1:  # 测试集
        padding_data = autograd.Variable(torch.zeros((len(instance_data), len_max))).long()
        padding_label = autograd.Variable(torch.zeros((len(instance_data), len_max))).long()
        mask = autograd.Variable(torch.zeros((len(instance_data), len_max))).long()
    else:
        padding_data = autograd.Variable(torch.zeros((batch_size, len_max))).long()
        padding_label = autograd.Variable(torch.zeros((batch_size, len_max))).long()
        mask = autograd.Variable(torch.zeros((batch_size, len_max))).long()
    len_list = torch.LongTensor(list(map(len, instance_data)))
    for idx, (seq, lab, len1) in enumerate(zip(instance_data, instance_label, len_list)):
        padding_data[idx, :len1] = torch.LongTensor(seq)
        padding_label[idx, :len1] = torch.LongTensor(lab)
        mask[idx, :len1] = torch.Tensor([1] * len1)

    padding_data = padding_data.permute(1, 0)
    padding_label = padding_label.permute(1, 0)
    if use_gpu:
        padding_data = padding_data.cuda()
        padding_label = padding_label.cuda()
    return padding_data, padding_label, mask


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):  # vec是1*5, type是Variable

    max_score = vec[0, argmax(vec)]
    # max_score维度是１，　max_score.view(1,-1)维度是１＊１，max_score.view(1, -1).expand(1, vec.size()[1])的维度是１＊５
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])  # vec.size()维度是1*5
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))  # 为什么指数之后再求和，而后才log呢


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j. 居然是随机初始化的！！！！！！！！！！！！！！！之后的使用也是用这随机初始化的值进行操作！！
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        if use_gpu:
            return (autograd.Variable(torch.randn(2, 32, self.hidden_dim // 2)).cuda(),
                    autograd.Variable(torch.randn(2, 32, self.hidden_dim // 2)).cuda())
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def init_hidden_test(self):
        if use_gpu:
            return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)).cuda(),
                    autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)).cuda())
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

        # 预测序列的得分

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(1, self.tagset_size).fill_(-10000.)  # 1*5 而且全是-10000

        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[
            START_TAG]] = 0.  # 因为start tag是4，所以tensor([[-10000., -10000., -10000.,      0., -10000.]])，将start的值为零，表示开始进行网络的传播，

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)  # 初始状态的forward_var，随着step t变化
        if use_gpu:
            forward_var = forward_var.cuda()
        n = 0

        # Iterate through the sentence 会迭代feats的行数次，
        for feat in feats:  # feat的维度是５ 依次把每一行取出来~
            # n += 1
            # if not n%500:
            #   print('feats进度:', n, '/', len(feats))
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):  # next tag 就是简单 i，从0到len
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1,
                                                               self.tagset_size)  # 维度是1*5 噢噢！原来，LSTM后的那个矩阵，就被当做是emit score了

                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)  # 维度是１＊５
                if use_gpu:
                    trans_score = trans_score.cuda()
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                # 第一次迭代时理解：
                # trans_score所有其他标签到Ｂ标签的概率
                # 由lstm运行进入隐层再到输出层得到标签Ｂ的概率，emit_score维度是１＊５，5个值是相同的
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).unsqueeze(0))
            # 此时的alphas t 是一个长度为5，例如<class 'list'>: [tensor(0.8259), tensor(2.1739), tensor(1.3526), tensor(-9999.7168), tensor(-0.7102)]
            forward_var = torch.cat(alphas_t).view(1, -1)  # 到第（t-1）step时５个标签的各自分数
        terminal_var = forward_var + self.transitions[self.tag_to_ix[
            STOP_TAG]]  # 最后只将最后一个单词的forward var与转移 stop tag的概率相加 tensor([[   21.1036,    18.8673,    20.7906, -9982.2734, -9980.3135]])
        alpha = log_sum_exp(terminal_var)  # alpha是一个0维的tensor

        return alpha

    def _forward_alg_new_parallel(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([feats.shape[0], self.tagset_size], -10000.)  # .to('cuda')
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.
        init_alphas = autograd.Variable(init_alphas)
        if use_gpu:
            init_alphas = init_alphas.cuda()

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list = []
        forward_var_list.append(init_alphas)

        for feat_index in range(feats.shape[1]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1)
            # gamar_r_l = torch.transpose(gamar_r_l,0,1)
            t_r1_k = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)  # +1
            # t_r1_k = feats[:,feat_index,:].repeat(feats.shape[0],1,1).transpose(1, 2)
            if use_gpu:
                self.transitions = self.transitions.cuda()
            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0)
            # forward_var_list.append(log_add(aa))
            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]].repeat([feats.shape[0], 1])
        # terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    # 得到feats
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        if len(sentence[0]) == 1:
            self.hidden = self.init_hidden_test()
        # embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        embeds = self.word_embeds(sentence)

        # embeds = embeds.unsqueeze(1)

        lstm_out, self.hidden = self.lstm(embeds, self.hidden)  # 11*1*4
        # lstm_out = lstm_out.view(sentence.size()[0] * sentence.size()[1], self.hidden_dim)  # 11*4

        lstm_feats = self.hidden2tag(lstm_out)  # 11*5 is a linear layer
        lstm_feats = lstm_feats.permute(1, 0, 2)

        return lstm_feats

    # 得到gold_seq tag的score 即根据真实的label 来计算一个score，但是因为转移矩阵是随机生成的，故算出来的score不是最理想的值
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence #feats 11*5  tag 11 维
        score = autograd.Variable(torch.Tensor([0]).cuda())
        tags = torch.cat(
            [torch.LongTensor([self.tag_to_ix[START_TAG]]).cuda(), tags])  # 将START_TAG的标签３拼接到tag序列最前面，这样tag就是12个了

        for i, feat in enumerate(feats):
            # self.transitions[tags[i + 1], tags[i]] 实际得到的是从标签i到标签i+1的转移概率
            # feat[tags[i+1]], feat是step i 的输出结果，有５个值，对应B, I, E, START_TAG, END_TAG, 取对应标签的值
            # transition【j,i】 就是从i ->j 的转移概率值
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _score_sentence_parallel(self, feats, tags):
        # Gives the score of provided tag sequences
        # feats = feats.transpose(0,1)
        tags = tags.permute(1, 0)

        score = torch.zeros(tags.shape[0])  # .to('cuda')
        if use_gpu:
            score = score.cuda()

        # tags = torch.cat([torch.full([tags.shape[0],1],self.tag_to_ix[START_TAG]).long(),tags],dim=1)
        if use_gpu:
            tags = torch.cat([torch.full([tags.shape[0], 1], self.tag_to_ix[START_TAG]).long().cuda(), tags], dim=1)
        else:
            tags = torch.cat([torch.full([tags.shape[0], 1], self.tag_to_ix[START_TAG]).long(), tags], dim=1)

        for i in range(feats.shape[1]):
            feat = feats[:, i, :]
            # print(feat.device)
            # print(tags.device)
            # print(score.device)
            # print('feat:', feat.size())
            # print('score:', score.size())
            # print('self.transitions[tags[:,i + 1], tags[:,i]]:', self.transitions[tags[:,i + 1], tags[:,i]].size())
            # print('feat[range(feat.shape[0]),tags[:,i + 1]]:', feat[range(feat.shape[0]),tags[:,i + 1]].size())
            # exit()
            score = score + \
                    self.transitions[tags[:, i + 1], tags[:, i]] + feat[range(feat.shape[0]), tags[:, i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[:, -1]]
        return score

    # 解码，得到预测的序列，以及预测序列的得分
    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]  # 其他标签（B,I,E,Start,End）到标签next_tag的概率
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)  # 从step0到step(i-1)时5个序列中每个序列的最大score
            backpointers.append(bptrs_t)  # bptrs_t有５个元素

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]  # 其他标签到STOP_TAG的转移概率
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):  # 从后向前走，找到一个best路径
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()  # 把从后向前的路径正过来
        return path_score, best_path

    def _viterbi_decode_new(self, feats):
        backpointers = []
        feats = feats.squeeze(0)
        # print(feats.size())

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)  # .to('cuda')
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var_list = []
        forward_var_list.append(init_vvars)
        # print(forward_var_list)

        for feat_index in range(feats.shape[0]):
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            # print(gamar_r_l.size())
            gamar_r_l = torch.squeeze(gamar_r_l)
            # print(gamar_r_l.size())
            # exit()
            if use_gpu:
                gamar_r_l = gamar_r_l.cuda()
            next_tag_var = gamar_r_l + self.transitions
            # bptrs_t=torch.argmax(next_tag_var,dim=0)
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbivars_t, 0) + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        # Transition to STOP_TAG
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        # print(best_path)
        start = best_path.pop()
        # print(start)
        # print(best_path)
        # exit()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)  # 11*5 经过了LSTM+Linear矩阵后的输出，之后作为CRF的输入。
        # print("feats:", feats.size)
        forward_score = self._forward_alg_new_parallel(feats)  # 0维的一个得分，20.*来着
        # print("forward_score:", forward_score)
        gold_score = self._score_sentence_parallel(feats, tags)  # tensor([ 4.5836])

        return torch.sum(forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode_new(lstm_feats)
        return score, tag_seq


START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 20
HIDDEN_DIM = 40
use_gpu = True

# Make up some training data
# training_data = [("the wall street journal reported today that apple corporation made money".split(),
#                   "B I I I O O O B I O O".split()),
#                  ("georgia tech is a university in georgia".split(), "B I O O O O B".split())]

training_data = [[[], []]]

sentence_id = 0
juhao_num = 0
with open('/content/gdrive/My Drive/LatticeLSTM-master3/ResumeNER/train.char.bmes', 'r', encoding='utf-8') as f:
    list1 = f.readlines()
    for idx in range(len(list1)):
        i = list1[idx]
        if len(i) - 1:
            if i.split()[0] == '。':
                juhao_num += 1
                if juhao_num == 5:
                    training_data.append([])
                    training_data[-1].append([])
                    training_data[-1].append([])
                    sentence_id += 1
                    juhao_num = 0
            training_data[sentence_id][0].append(i.split()[0])
            training_data[sentence_id][1].append(i.split()[-1])

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

# tag_to_ix = {"B-NAME": 0, "E-NAME": 1, "O": 2, "B-CONT": 3, "M-CONT": 4, "E-CONT": 5, "B-RACE": 6, "E-RACE": 7,
#              "B-TITLE": 8, "M-TITLE": 9, "E-TITLE": 10, "B-EDU": 11, "M-EDU": 12, "E-EDU": 13, "B-ORG": 14, "M-ORG": 15,
#              "E-ORG": 16, "M-NAME": 17, "B-PRO": 18, "M-PRO": 19, "E-PRO": 20, "S-RACE": 21, "S-NAME": 22, "B-LOC": 23,
#              "M-LOC": 24, "E-LOC": 25, "M-RACE": 26, "S-ORG": 27, START_TAG: 28, STOP_TAG: 29}

tag_to_ix = {"O": 0, "B-NAME": 1, "M-NAME": 2, "E-NAME": 3, "B-CONT": 4, "M-CONT": 5, "E-CONT": 6, "B-RACE": 7, "M-RACE": 8,
             "E-RACE": 9, "B-TITLE": 10, "M-TITLE": 11, "E-TITLE": 12, "B-EDU": 13, "M-EDU": 14, "E-EDU": 15, "B-ORG": 16, "M-ORG": 17,
             "E-ORG": 18,  "B-PRO": 19, "M-PRO": 20, "E-PRO": 21, "B-LOC": 22,
             "M-LOC": 23, "E-LOC": 24, "S-NAME": 25, "S-RACE": 26, "S-ORG": 27, START_TAG: 28, STOP_TAG: 29}
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.0001, weight_decay=1e-4)
if use_gpu:
    model = model.cuda()
    # optimizer = optimizer.cuda()

# Check predictions before training
# precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
# precheck_tags = torch.LongTensor([tag_to_ix[t] for t in training_data[0][1]])
# print(model(precheck_sent))

idx = 1
for sentence, tags in training_data:
    sentence_in = prepare_sequence(sentence, word_to_ix)
    training_data[idx - 1][0] = sentence_in
    # targets = torch.LongTensor([tag_to_ix[t] for t in tags])
    targets = [tag_to_ix[t] for t in tags]
    training_data[idx - 1][1] = targets
    idx += 1

# test_data = [("刘 强 东 在 北 京 向 马 云 学 习".split(), "B-NAME M-NAME E-NAME O B-LOC E-LOC O B-NAME E-NAME O O".split())]

test_data = [[[], []]]
with open('/content/gdrive/My Drive/LatticeLSTM-master3/ResumeNER/test.char.bmes', 'r', encoding='utf-8') as f:
    list1 = f.readlines()
    juhao_num1 = 0
    char_num = len(list1)
    for idxx in range(char_num):
        i = list1[idxx]
        if len(i) - 1:
            if i.split()[0] == '。':
                juhao_num1 += 1
            if juhao_num1 == 14:
                pass
            elif juhao_num1 == 17:
                pass
            elif juhao_num1 == 21:
                pass
            elif juhao_num1 == 22:
                pass
            elif juhao_num1 == 36:
                pass
            elif juhao_num1 == 40:
                pass
            elif juhao_num1 == 42:
                pass
            elif juhao_num1 == 46:
                pass
            elif juhao_num1 == 52:
                pass
            else:
                test_data[0][0].append(i.split()[0])
                test_data[0][1].append(i.split()[-1])
        if juhao_num1 == 70:
            break

precheck_sent = prepare_sequence(test_data[0][0], word_to_ix)
precheck_tags = prepare_sequence(test_data[0][1], tag_to_ix)
precheck_sent = [precheck_sent]
precheck_tags = [precheck_tags]
pad_1, pad_2, pad_3 = padding(precheck_sent, precheck_tags)

batch_size = 32
batch_num = idx // batch_size + 1

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(100):  # again, normally you would NOT do 300 epochs, it is toy data
    print('epoch:', epoch + 1)
    print('batch_num:', batch_num)
    random.shuffle(training_data)

    for batch_id in range(batch_num):
        model.zero_grad()
        # print('batch_id:', batch_id)
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > idx:
            end = idx

        instance = training_data[start:end]
        instance_data = [sent[0] for sent in instance]
        instance_label = [sent[1] for sent in instance]

        padding_data, padding_label, mask = padding(instance_data, instance_label)

        neg_log_likelihood = model.neg_log_likelihood(padding_data,
                                                      padding_label)  # tensor([ 15.4958]) 最大的可能的值与 根据随机转移矩阵 计算的真实值 的差

        print('batch_id: %s , neg_log_likelihood: %s' % (batch_id, neg_log_likelihood))
        # print('neg_log_likelihood:', neg_log_likelihood)
        neg_log_likelihood.backward()
        optimizer.step()
    tag_predict = model(pad_1)[1]
    print('tag_predict:', tag_predict)  # tag sequence
    tag_gold = precheck_tags[0]
    print('tags_gold:', tag_gold)
    assert len(tag_predict) == len(tag_gold)
    n0 = 0
    n1 = 0
    n2 = 0
    for id_tag in range(len(tag_gold)):
        if tag_gold[id_tag] != 2:
            n0 += 1
        if tag_predict[id_tag] == tag_gold[id_tag]:
            n1 += 1
            if tag_predict[id_tag] != 2:
                n2 += 1
    print('第%s轮训练：总共%s个字，正确%s个字，正确率为%.5s%%， 抛去“O”标签，总共%s个字，正确%s个字，正确率为%.5s%%' % (
    epoch + 1, len(tag_gold), n1, n1 * 100 / len(tag_gold), n0, n2, n2 * 100 / n0))

# 12月9日改动:   1.训练时主部分改成batch_num
#               2.初始向量(2, 32, hidden//2)
#               3.注释了 embeds = embeds.unsqueeze(1)
#               4.  先torch.autograd.Variable再包起来 变成 先包起来再torch.autograd.Variable
#               5. 改动def prepare_sequence
#               6. def _get_lstm_features里 len()改成.size()[0]