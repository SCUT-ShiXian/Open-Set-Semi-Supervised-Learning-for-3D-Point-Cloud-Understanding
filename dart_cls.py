import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

from util import cal_loss, IOStream
import torch.nn.functional as F

meta_label2seg = {'0':[0, 1, 2, 3],
         '1':[4, 5, 6, 7],
         '2':[8, 9, 10, 11],
         '3':[12, 13, 14],
         '4':[15, 16],
         '5':[17, 18, 19, 20],
         '6':[21, 22],
         '7':[23, 24, 25]
         }

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy()]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

class Architect(object):
  def __init__(self, model, predictor, predictor_teacher, prd_opt, num_classes, args):
    self.network_momentum = args.prd_momentum
    self.network_weight_decay = args.arch_wd
    self.model = model
    self.predictor = predictor
    self.predictor_teacher = predictor_teacher
    
    self.optimizer = prd_opt

    self.num_classes = num_classes
    self.time = 0

  def _compute_unrolled_model(self, input, label, input_valid, batch_ids, eta, network_optimizer, w_etp):
    ### compute loss
    criterion_mse = torch.nn.MSELoss(reduction='sum').cuda()
    logits, feat = self.model(input)
    logits = logits.contiguous()

    with torch.no_grad():
      weights = self.predictor(feat)
    weight_watch = weights.reshape(-1, 1)
    sfm_pred = torch.softmax(logits, dim=1)
    loss_prd = criterion_mse(weight_watch*sfm_pred, weight_watch*label) / float(logits.shape[0])

    loss = loss_prd

    theta = _concat(self.model.parameters()).data.detach()
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data.detach() + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub_(moment+dtheta, alpha=eta))
    return unrolled_model


  def step(self, input_train, label_train, input_valid, label_valid, weights_ema, batch_ids, eta, network_optimizer, w_etp):
    self.optimizer.zero_grad()

    prd_grads = []
    nums_val = len(input_valid)
    for i_inp in range(nums_val):
        # self.optimizer.zero_grad()
        one_input_val = input_valid[i_inp]
        one_label_val = label_valid[i_inp]
        unrolled_loss, loss_cos, loss_binaryU, loss_binaryX = self._backward_step_unrolled(input_train, label_train, one_input_val, one_label_val, weights_ema, batch_ids, eta, network_optimizer, w_etp)
        if len(prd_grads) == 0:
            for v in self.predictor.parameters():
                prd_grads.append(v.grad.data / nums_val)
        else:
            for v, g in zip(self.predictor.parameters(), prd_grads):
                g.data.copy_(g.data + v.grad.data / nums_val)
        # self.optimizer.step()

    for v, g in zip(self.predictor.parameters(), prd_grads):
        v.grad.data.copy_(g.data)


    # #### clip grad
    for p in self.predictor.parameters():
        nn.utils.clip_grad_norm_(p, 1)
    self.optimizer.step()
    return unrolled_loss, loss_cos, loss_binaryU, loss_binaryX

  def _backward_step_unrolled(self, input_train, label_train, input_valid, label_valid, weights_ema, batch_ids, eta, network_optimizer, w_etp):
    unrolled_model = self._compute_unrolled_model(input_train, label_train, input_valid, batch_ids, eta, network_optimizer, w_etp)
    
    logits, feat = unrolled_model(input_train)
    logits = logits.contiguous()
    logits_val, feat_val = unrolled_model(input_valid)
    logits_val = logits_val.contiguous()
    feat = feat.detach()
    feat_val = feat_val.detach()

    unrolled_loss = 1.0 * F.cross_entropy(logits_val, label_valid, reduction='mean')


    weights = self.predictor(feat)
    # weights = torch.clamp(weights, min=0.0, max=2.0)
    weights_val = self.predictor(feat_val)


    loss_cos = 0.01*torch.sum(torch.linalg.norm(weights - weights_ema, dim=-1) / torch.linalg.norm(weights_ema + 1e-6, dim=-1)) / float(logits.shape[0])
    if w_etp>0:
      loss_binaryU = w_etp*0.01*torch.sum(-weights * torch.log(weights+1e-6) - (1.0-weights) * torch.log(1.0-weights+1e-6)) / float(logits.shape[0])
    else:
      loss_binaryU = 0.01*torch.sum(-torch.log(1.0-weights+1e-6)) / float(logits.shape[0])
    loss_binaryX = 0.01*torch.sum(-torch.log(weights_val)) / float(logits_val.shape[0])

    #### regular loss and outlier loss
    loss = unrolled_loss + loss_cos + loss_binaryU + loss_binaryX

    loss.backward()
    dalpha = []
    vector = [v.grad.data.detach() for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, label_train, batch_ids)
    for ig in implicit_grads:
        dalpha += [-ig]

    for v, g in zip(self.predictor.parameters(), dalpha):
        if v.grad is None:
            if not (g is None):
                v.grad = Variable(g.data)
        else:
            if not (g is None):
                v.grad.data.copy_(v.grad.data + g.data)
    

    return unrolled_loss, loss_cos, loss_binaryU, loss_binaryX

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length


    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()




  def _hessian_vector_product(self, vector, input, label, batch_ids, r=1e-2):
    criterion_mse = torch.nn.MSELoss(reduction='sum').cuda()

    R = r / _concat(vector).data.detach().norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    ### compute loss
    with torch.no_grad():
      logits, feat = self.model(input)
      logits = logits.contiguous()

    weights = self.predictor(feat)
    weight_watch = weights.reshape(-1, 1)
    sfm_pred = torch.softmax(logits, dim=1)
    loss = criterion_mse(weight_watch*sfm_pred, weight_watch*label) / float(logits.shape[0])

    grads_p = torch.autograd.grad(loss, self.predictor.parameters(), retain_graph=True, allow_unused=True)
    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)


    ### compute loss  
    with torch.no_grad():
      logits, feat = self.model(input)
      logits = logits.contiguous()
    weights = self.predictor(feat)
    weight_watch = weights.reshape(-1, 1)
    sfm_pred = torch.softmax(logits, dim=1)
    loss = criterion_mse(weight_watch*sfm_pred, weight_watch*label) / float(logits.shape[0])


    grads_n = torch.autograd.grad(loss, self.predictor.parameters(), retain_graph=True, allow_unused=True)
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


