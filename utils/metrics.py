import logging

import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from torchmetrics.text.bert import BERTScore

logger = logging.getLogger(__name__)


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean()


def compute_caption_metric(ground_truth, prediction):
    """
    Args:
        ground_truth: a list of ground truth captions
        prediction: a list of predicted captions
    """
    assert len(ground_truth) == len(prediction), 'groud truth and prediction must be aligned.'
    gts, res = {}, {}
    for ind, (gt, pred) in enumerate(zip(ground_truth, prediction)):
        gts[ind] = [{'caption': gt}]
        res[ind] = [{'caption': pred}]

    tokenizer = PTBTokenizer()
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    report = {}
    for metric, method in scorers:
        scores, _ = metric.compute_score(
            tokenizer.tokenize(gts),
            tokenizer.tokenize(res)
        )
        if isinstance(method, list):
            for m, s in zip(method, scores):
                report[m] = s
        else:
            report[method] = scores
    # BERT score ("We compute systemlevel scores by averaging BERTSCORE for every reference-candidate pair." from the paper)
    bertscore = BERTScore()
    score = bertscore(ground_truth, prediction)
    for k, v in score.items():
        report[f'BERTScore_{k}'] = sum(v)/len(v)
    return report
