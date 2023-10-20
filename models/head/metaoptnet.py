import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from qpth.qp import QPFunction
from torch.autograd import Variable

from models.model import register
from utils import make_nk_label

logger = logging.getLogger(__name__)


@register('metaoptnet')
class MetaOptNet(nn.Module):

    def __init__(self, head='SVM', normalize=True):
        super().__init__()
        self.head = head
        self.normalize = normalize

        # Choose the classification head
        if self.head == 'ProtoNet':
            logger.info('Method: MetaOptNet, head: ProtoNet, Normalize: {}'.format(self.normalize))
            self.cls_head = ClassificationHead(base_learner='ProtoNet')
        elif self.head == 'SVM':
            logger.info('Method: MetaOptNet, head: SVM')
            self.cls_head = ClassificationHead(base_learner='SVM-CS')
        else:
            logger.info("Cannot recognize the dataset type")
            assert (False)

    def forward(self, x_shot_feat, x_query_feat, batch_size, n_way, n_shot, n_query):
        query_shape = [batch_size, n_query]

        x_shot = x_shot_feat.view(batch_size, n_way * n_shot, -1)  # [bs, n_way * n_shot, n_feat]
        x_query = x_query_feat.view(*query_shape, -1)  # [bs, n_query, n_feat]

        labels_support = make_nk_label(n_way, n_shot, batch_size)
        labels_support = labels_support.view(batch_size, -1)

        logits = self.cls_head(x_query, x_shot, labels_support, n_way, n_shot,
                               normalize=self.normalize)

        return logits


class ClassificationHead(nn.Module):
    def __init__(self, base_learner='MetaOptNet', enable_scale=True):
        super(ClassificationHead, self).__init__()
        if 'SVM-CS' in base_learner:
            self.head = MetaOptNetHead_SVM_CS
        elif 'Proto' in base_learner:
            self.head = ProtoNetHead
        else:
            logger.error("Cannot recognize the base learner type")
            assert (False)

        # Add a learnable scale
        self.enable_scale = enable_scale
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))

    def forward(self, query, support, support_labels, n_way, n_shot, **kwargs):
        if self.enable_scale:
            return self.scale * self.head(query, support, support_labels, n_way, n_shot, **kwargs)
        else:
            return self.head(query, support, support_labels, n_way, n_shot, **kwargs)


def MetaOptNetHead_SVM_CS(query, support, support_labels, n_way, n_shot,
                          C_reg=0.1, double_precision=False, maxIter=15, **kwargs):
    """
    Fits the support set with multi-class SVM and
    returns the classification score on the query set.

    This is the multi-class SVM presented in:
    On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
    (Crammer and Singer, Journal of Machine Learning Research 2001).

    This model is the classification head that we use for the final version.
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """

    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

    # Here we solve the dual problem:
    # Note that the classes are indexed by m & samples are indexed by i.
    # min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
    # s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

    # where w_m(\alpha) = \sum_i \alpha^m_i x_i,
    # and C^m_i = C if m  = y_i,
    # C^m_i = 0 if m != y_i.
    # This borrows the notation of liblinear.

    # \alpha is an (n_support, n_way) matrix
    kernel_matrix = computeGramMatrix(support, support)

    id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).to(query.device)
    block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
    # This seems to help avoid PSD error from the QP solver.
    block_kernel_matrix += 1.0 * torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support,
                                                                     n_way * n_support).to(query.device)

    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support),
                                     n_way).to(query.device)  # (tasks_per_batch * n_support, n_support)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * n_way)

    G = block_kernel_matrix
    e = -1.0 * support_labels_one_hot
    # logger.info (G.size())
    # This part is for the inequality constraints:
    # \alpha^m_i <= C^m_i \forall m,i
    # where C^m_i = C if m  = y_i,
    # C^m_i = 0 if m != y_i.
    id_matrix_1 = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support).to(query.device)
    C = Variable(id_matrix_1)
    h = Variable(C_reg * support_labels_one_hot)
    # logger.info (C.size(), h.size())
    # This part is for the equality constraints:
    # \sum_m \alpha^m_i=0 \forall i
    id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).to(query.device)

    A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).to(query.device)))
    b = Variable(torch.zeros(tasks_per_batch, n_support).to(query.device))
    # logger.info (A.size(), b.size())
    if double_precision:
        G, e, C, h, A, b = [x.double() for x in [G, e, C, h, A, b]]
    else:
        G, e, C, h, A, b = [x.float() for x in [G, e, C, h, A, b]]

    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    with torch.cuda.amp.autocast(enabled=False):
        qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())

    # Compute the classification score.
    compatibility = computeGramMatrix(support, query)
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
    qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
    logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
    logits = logits * compatibility
    logits = torch.sum(logits, 1)

    return logits


def ProtoNetHead(query, support, support_labels, n_way, n_shot, normalize=True, **kwargs):
    """
    Constructs the prototype representation of each class(=mean of support vectors of each class) and
    returns the classification score (=L2 distance to each class prototype) on the query set.

    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      normalize: a boolean. Represents whether if we want to normalize the distances by the embedding dimension.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """

    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)
    d = query.size(2)

    assert (query.dim() == 3)
    assert (support.dim() == 3)
    assert (query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert (n_support == n_way * n_shot)  # n_support must equal to n_way * n_shot

    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way).to(query.device)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)

    # ************************* Compute Prototypes **************************
    labels_train_transposed = support_labels_one_hot.transpose(1, 2)
    # Batch matrix multiplication:
    #   prototypes = labels_train_transposed * features_train ==>
    #   [batch_size x nKnovel x num_channels] =
    #       [batch_size x nKnovel x num_train_examples] * [batch_size * num_train_examples * num_channels]
    prototypes = torch.bmm(labels_train_transposed, support)
    # Divide with the number of examples per novel category.
    prototypes = prototypes.div(
        labels_train_transposed.sum(dim=2, keepdim=True).expand_as(prototypes)
    )

    # Distance Matrix Vectorization Trick
    AB = computeGramMatrix(query, prototypes)
    AA = (query * query).sum(dim=2, keepdim=True)
    BB = (prototypes * prototypes).sum(dim=2, keepdim=True).reshape(tasks_per_batch, 1, n_way)
    logits = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB)
    logits = -logits



    if normalize:
        logits = logits / d

    return logits


def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.

    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """

    assert (A.dim() == 3)
    assert (B.dim() == 3)
    assert (A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1, 2))


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape(
        [matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(
        matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))
