import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from util import cal_loss, IOStream

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


class Architect(object):

  def __init__(self, model, predictor, predictor_teacher, prd_opt, num_classes, category_seg_num, args):
    self.network_momentum = args.prd_momentum
    self.network_weight_decay = args.arch_wd
    self.model = model
    self.predictor = predictor
    self.predictor_teacher = predictor_teacher

    self.optimizer = prd_opt
    self.num_classes = num_classes
    self.category_seg_num = category_seg_num

  def _compute_unrolled_model(self, input, target, label, batch_ids, eta, network_optimizer):
    criterion_mse = nn.MSELoss(reduction='sum').cuda()
    logits, feat = self.model(input.permute(0, 2, 1), to_categorical(label, self.num_classes))
    logits = logits.contiguous()

    with torch.no_grad():
      weights = self.predictor(feat)
    weight_watch = weights.reshape(input.shape[0], 1, 1)
    #### soft unsupervised loss
    sfm_pred = torch.softmax(logits, dim=1)
    loss = criterion_mse(weight_watch*sfm_pred, weight_watch*target) / float(input.shape[0] * input.shape[1])

    theta = _concat(self.model.parameters()).data.detach()

    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data.detach() + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub_(moment+dtheta, alpha=eta))
    return unrolled_model


  def step(self, input_train, target_train, label_train, input_valid, target_valid, label_valid, batch_ids, eta, network_optimizer):
    self.optimizer.zero_grad()
    unrolled_loss, loss_cos, loss_binaryU, loss_binaryX = self._backward_step_unrolled(input_train, target_train, label_train, input_valid, target_valid, label_valid, batch_ids, eta, network_optimizer)

    self.optimizer.step()
    return unrolled_loss, loss_cos, loss_binaryU, loss_binaryX


  def _backward_step_unrolled(self, input_train, target_train, label_train, input_valid, target_valid, label_valid, batch_ids, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, label_train, batch_ids, eta, network_optimizer)    

    logits, feat = unrolled_model(input_train.permute(0, 2, 1), to_categorical(label_train, self.num_classes))
    logits = logits.permute(0, 2, 1).contiguous()
    logits_val, feat_val = unrolled_model(input_valid.permute(0, 2, 1), to_categorical(label_valid, self.num_classes))
    logits_val = logits_val.permute(0, 2, 1).contiguous()
    feat = feat.detach()
    feat_val = feat_val.detach()


    unrolled_loss = 10.0*cal_loss(logits_val.view(-1, logits_val.shape[2]), target_valid.view(-1,1).squeeze(), smoothing=False)

    weights = self.predictor(feat)
    weights_val = self.predictor(feat_val)
    with torch.no_grad():
        weights_ema = self.predictor_teacher(feat)

    loss_cos = 0.1*torch.sum(torch.linalg.norm(weights - weights_ema, dim=-1) / torch.linalg.norm(weights_ema + 1e-6, dim=-1)) / float(logits.shape[0])
    loss_binaryU = 1.0*torch.sum(-torch.log(1.0-weights+1e-6)) / float(logits.shape[0])
    loss_binaryX = 1.0*torch.sum(-torch.log(weights_val)) / float(logits_val.shape[0])

    loss = unrolled_loss + loss_cos + loss_binaryU + loss_binaryX
    loss.backward()

    dalpha = []
    vector = [v.grad.data.detach() for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train, label_train, batch_ids)

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




  def _hessian_vector_product(self, vector, input, target, label, batch_ids, r=1e-2):
    criterion_mse = nn.MSELoss(reduction='sum').cuda()

    R = r / _concat(vector).data.detach().norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    with torch.no_grad():
      logits, feat = self.model(input.permute(0, 2, 1), to_categorical(label, self.num_classes))
      logits = logits.contiguous()
    weights = self.predictor(feat)
    weight_watch = weights.reshape(input.shape[0], 1, 1)
    #### soft unsupervised loss
    sfm_pred = torch.softmax(logits, dim=1)
    loss = criterion_mse(weight_watch*sfm_pred, weight_watch*target) / float(input.shape[0] * input.shape[1])

    grads_p = torch.autograd.grad(loss, self.predictor.parameters(), retain_graph=True, allow_unused=True)
    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)

    with torch.no_grad():
      logits, feat = self.model(input.permute(0, 2, 1), to_categorical(label, self.num_classes))
      logits = logits.contiguous()
    weights = self.predictor(feat)
    weight_watch = weights.reshape(input.shape[0], 1, 1)
    #### soft unsupervised loss
    sfm_pred = torch.softmax(logits, dim=1)
    loss = criterion_mse(weight_watch*sfm_pred, weight_watch*target) / float(input.shape[0] * input.shape[1])

    grads_n = torch.autograd.grad(loss, self.predictor.parameters(), retain_graph=True, allow_unused=True)
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]


