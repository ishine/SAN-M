from numpy import dtype
import oneflow.experimental as flow
import oneflow.experimental.nn as nn

from model.pad_mask_utils import IGNORE_ID


def cal_performance(pred, gold, smoothing=0.0):      

    pred = pred.reshape(shape=[-1, pred.size(2)])                  
    gold = gold.reshape(shape=[-1])                 

    loss = cal_loss(pred, gold, smoothing)

    pred = pred.max(1)[1]
    non_pad_mask = gold.ne(IGNORE_ID)
    n_correct = pred.eq(gold)
    n_correct = n_correct.masked_select(non_pad_mask).sum().item()

    return loss, n_correct


def cal_loss(pred, gold, smoothing=0.0):
    if smoothing > 0.0:
        eps = smoothing
        n_class = pred.size(1)

        #gold_for_scatter = gold.ne(IGNORE_ID).long() * gold
        gold_for_scatter = gold.ne(IGNORE_ID).to(dtype=flow.int32)*gold
        print(flow.zeros_like(pred))
        one_hot = flow.zeros_like(pred).scatter(1, gold_for_scatter.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / n_class
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = gold.ne(IGNORE_ID)
        n_word = non_pad_mask.sum().item()
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).sum() / n_word
    else:
        loss = F.cross_entropy(pred, gold,
                               ignore_index=IGNORE_ID,
                               reduction='elementwise_mean')

    return loss
