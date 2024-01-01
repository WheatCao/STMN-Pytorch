from __future__ import print_function
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from numpy.testing import assert_array_almost_equal


def loss_transfer(model, loss, batch_size):

    metrics = []
    for i in range(batch_size):
        loss_ = loss[i]
        loss_.backward(retain_graph=True)
        to_concat_g = []
        to_concat_v = []
        for name, param in model.named_parameters():
            if param.dim() in [2, 4]:
                to_concat_g.append(param.grad.data.view(-1))
                to_concat_v.append(param.data.view(-1))
        all_g = torch.cat(to_concat_g)
        all_v = torch.cat(to_concat_v)
        metric = torch.abs(torch.sum(all_g * all_v))
        model.zero_grad()
        metrics.append(metric)
    metrics = np.array(metrics)
    return metrics


def loss_single(model, y_1, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    g_1 = loss_transfer(model, loss_1, len(loss_1))
    # print(g_1)
    ind_1_sorted = np.argsort(g_1)
    loss_1_sorted = loss_1[ind_1_sorted]

    remember = 1 - forget_rate
    num_remember = int(remember * len(loss_1_sorted))

    ind_1_update = ind_1_sorted[:num_remember]

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_update]])/float(num_remember)

    loss_1_update = F.cross_entropy(y_1[ind_1_update], t[ind_1_update])
    
    return torch.sum(loss_1_update)/num_remember, pure_ratio_1, ind_1_update




# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduction='none')
    ind_1_sorted = np.argsort(loss_1.cpu().data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduction='none')
    ind_2_sorted = np.argsort(loss_2.cpu().data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    ind_1_update=ind_1_sorted[:num_remember].cpu()
    ind_2_update=ind_2_sorted[:num_remember].cpu()
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted.cpu().numpy()
        ind_2_update = ind_2_sorted.cpu().numpy()
        num_remember = ind_1_update.shape[0]

    pure_ratio_1 = 0.
    pure_ratio_2 = 0.

    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2

def loss_coteaching_plus(logits, logits2, labels, forget_rate, ind, noise_or_not, step):
    outputs = F.softmax(logits, dim=1)
    outputs2 = F.softmax(logits2, dim=1)

    _, pred1 = torch.max(logits.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id=np.zeros(labels.size(), dtype=bool)
    disagree_id = []
    for idx, p1 in enumerate(pred1): 
        if p1 != pred2[idx]:
            disagree_id.append(idx) 
            logical_disagree_id[idx] = True
    
    temp_disagree = ind*logical_disagree_id.astype(np.int64)
    ind_disagree = np.asarray([i for i in temp_disagree if i != 0]).transpose()
    try:
        assert ind_disagree.shape[0]==len(disagree_id)
    except:
        disagree_id = disagree_id[:ind_disagree.shape[0]]
     
    _update_step = np.logical_or(logical_disagree_id, step < 5000).astype(np.float32)
    update_step = Variable(torch.from_numpy(_update_step)).cuda()

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outputs = outputs[disagree_id] 
        update_outputs2 = outputs2[disagree_id] 
        
        loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(update_outputs, update_outputs2, update_labels, forget_rate, ind_disagree, noise_or_not)
    else:
        update_labels = labels
        update_outputs = outputs
        update_outputs2 = outputs2

        cross_entropy_1 = F.cross_entropy(update_outputs, update_labels)
        cross_entropy_2 = F.cross_entropy(update_outputs2, update_labels)

        loss_1 = torch.sum(update_step*cross_entropy_1)/labels.size()[0]
        loss_2 = torch.sum(update_step*cross_entropy_2)/labels.size()[0]
 
        pure_ratio_1 = 0.
        pure_ratio_2 = 0.
    return loss_1, loss_2, pure_ratio_1, pure_ratio_2

def kl_loss_compute(pred, soft_targets, reduce=True):

    kl = F.kl_div(F.log_softmax(pred, dim=1),F.softmax(soft_targets, dim=1),reduce=False)

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)




def loss_jocor(y_1, y_2, t, forget_rate, ind, noise_or_not, co_lambda=0.1):

    loss_pick_1 = F.cross_entropy(y_1, t, reduce = False) * (1-co_lambda)
    loss_pick_2 = F.cross_entropy(y_2, t, reduce = False) * (1-co_lambda)
    loss_pick = (loss_pick_1 + loss_pick_2 + co_lambda * kl_loss_compute(y_1, y_2,reduce=False) + co_lambda * kl_loss_compute(y_2, y_1, reduce=False)).cpu()


    ind_sorted = np.argsort(loss_pick.data)
    loss_sorted = loss_pick[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    pure_ratio = 0.

    ind_update=ind_sorted[:num_remember]

    # exchange
    loss = torch.mean(loss_pick[ind_update])

    return loss, loss, pure_ratio, pure_ratio


def loss_decoupling(logits1, logits2, labels, step):
    _, pred1 = torch.max(logits1.data, 1)
    _, pred2 = torch.max(logits2.data, 1)

    pred1, pred2 = pred1.cpu().numpy(), pred2.cpu().numpy()

    logical_disagree_id = np.zeros(labels.size(), dtype=bool)
    disagree_id=[]
    for idx, p1 in enumerate(pred1): 
        if p1 != pred2[idx]:
            disagree_id.append(idx) 
            logical_disagree_id[idx] = True
    
    int_logical_disagree_id = logical_disagree_id.astype(np.int64)
    nonzeros = np.nonzero(int_logical_disagree_id)
    nonzero_int_logical_disagree_id = int_logical_disagree_id[nonzeros]

    _update_step = np.logical_or(nonzero_int_logical_disagree_id, step<5000).astype(np.float32)
    update_step = Variable(torch.from_numpy(_update_step)).cuda()

    if len(disagree_id) > 0:
        update_logits1 = logits1[disagree_id]
        update_logits2 = logits2[disagree_id]
        update_labels = labels[disagree_id]
    else:
        update_logits1 = logits1
        update_logits2 = logits2
        update_labels = labels

    cross_entropy_1 = F.cross_entropy(update_logits1, update_labels)
    cross_entropy_2 = F.cross_entropy(update_logits2, update_labels)

    loss_1 = torch.sum(update_step*cross_entropy_1)/update_labels.size()[0]
    loss_2 = torch.sum(update_step*cross_entropy_2)/update_labels.size()[0]

    return loss_1, loss_2  


def loss_forget(logits, labels, forget_rate, ind, noise_or_not):
    loss = F.cross_entropy(logits, labels, reduction='none')
    ind_sorted = np.argsort(loss.cpu().data).cuda()
    loss_sorted = loss[ind_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_sorted))

    pure_ratio = 0.

    loss_small = loss_sorted[:num_remember]

    return torch.sum(loss_small)/num_remember, pure_ratio
