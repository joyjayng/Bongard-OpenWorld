import logging
import random

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import models
import utils
from models.misc.optimizer import make_optimizer

logger = logging.getLogger(__name__)


class BongardModel(pl.LightningModule):
    def __init__(self,
                 image_encoder_name,
                 image_encoder_output_dim,
                 image_encoder_args,
                 relational_encoder_name,
                 few_shot_head_name,
                 few_shot_head_args,
                 caption_head_name,
                 caption_head_args,
                 optimizer_name,
                 optimizer_args,
                 bongard_args,
                 freeze_image_encoder=False,
                 caption_loss_coeff=0.1,
                 additional_args=None):
        super().__init__()
        self.save_hyperparameters()

        # image encoder
        self.image_encoder = models.make(
            image_encoder_name, **image_encoder_args
        )
        self.freeze_image_encoder = freeze_image_encoder
        if self.freeze_image_encoder:
            self.image_encoder.eval()
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        # RN encoder
        if relational_encoder_name:
            self.relational_encoder = models.make(
                relational_encoder_name,
                input_feat_dim=image_encoder_output_dim,
            )
            shot_feat_dim = self.relational_encoder.out_dim
        else:
            self.relational_encoder = None
            shot_feat_dim = image_encoder_output_dim

        # FSL head
        if few_shot_head_name in ['wren', 'snail']:
            few_shot_head_args.input_feat_dim = shot_feat_dim
        self.fsl_head = models.make(
            few_shot_head_name, **few_shot_head_args
        )
        self.is_snail = True if 'snail' in few_shot_head_name else False

        # captioning head
        if caption_head_name:
            caption_head_args.input_feat_dim = shot_feat_dim
            self.caption_head_args = caption_head_args
            self.caption_head = models.make(
                caption_head_name, **caption_head_args
            )
        else:
            self.caption_head = None
            self.caption_head_args = None

        self.optimizer_name = optimizer_name
        self.optimizer_args = optimizer_args
        self.bongard_args = bongard_args
        self.caption_loss_coeff = caption_loss_coeff
        self.additional_args = additional_args

        # metric
        self.train_acc = utils.AverageMeter()
        self.train_acc_concept_2 = utils.AverageMeter()
        self.train_acc_concept_3 = utils.AverageMeter()
        self.train_acc_concept_4 = utils.AverageMeter()
        self.train_acc_concept_5 = utils.AverageMeter()
        self.train_acc_commonsense = utils.AverageMeter()
        self.train_acc_non_commonsense = utils.AverageMeter()
        self.val_acc = utils.AverageMeter()
        self.val_acc_concept_2 = utils.AverageMeter()
        self.val_acc_concept_3 = utils.AverageMeter()
        self.val_acc_concept_4 = utils.AverageMeter()
        self.val_acc_concept_5 = utils.AverageMeter()
        self.val_acc_commonsense = utils.AverageMeter()
        self.val_acc_non_commonsense = utils.AverageMeter()
        self.test_acc = utils.AverageMeter()
        self.test_acc_concept_2 = utils.AverageMeter()
        self.test_acc_concept_3 = utils.AverageMeter()
        self.test_acc_concept_4 = utils.AverageMeter()
        self.test_acc_concept_5 = utils.AverageMeter()
        self.test_acc_commonsense = utils.AverageMeter()
        self.test_acc_non_commonsense = utils.AverageMeter()
        self.test_acc_pos = utils.AverageMeter()
        self.test_acc_neg = utils.AverageMeter()

        self.viz_caption_buffer = []
        self.max_viz_caption = 5

        self.wrong_q = []
        # TBD: captioning metric

        self.automatic_optimization = False

    def _pred(self, batch, generate_text=False):
        B = batch['shot_ims'].size(0)
        shot_feat = self.image_encoder(rearrange(
            batch['shot_ims'], 'b n c h w -> (b n) c h w'))
        query_feat = self.image_encoder(rearrange(
            batch['query_ims'], 'b n c h w -> (b n) c h w'))

        # use rn encoder
        if self.relational_encoder is not None:
            shot_feat = self.relational_encoder(
                shot_feat,
                rearrange(batch['shot_boxes'], 'b n m c -> (b n) m c'),
                rearrange(batch['shot_norm_boxes'], 'b n m c -> (b n) m c'),
            )
            query_feat = self.relational_encoder(
                query_feat,
                rearrange(batch['query_boxes'], 'b n m c -> (b n) m c'),
                rearrange(batch['query_norm_boxes'], 'b n m c -> (b n) m c'),
            )
        # max pool
        else:
            original_shot_feat = shot_feat.reshape(*shot_feat.shape[:-2], -1) # (b n) c (h w)
            shot_feat = shot_feat.reshape(
                *shot_feat.shape[:-2], -1).max(-1).values
            query_feat = query_feat.reshape(
                *query_feat.shape[:-2], -1).max(-1).values

        # fsl head
        if self.is_snail:
            logits = []
            for i in range(self.bongard_args.n_query):
                logit = self.fsl_head(
                    rearrange(
                        shot_feat,
                        '(b n_way n_shot) c -> b (n_way n_shot) c',
                        b=B,
                        n_shot=self.bongard_args.n_shot,
                        n_way=self.bongard_args.n_way),
                    rearrange(
                        query_feat, '(b n_query) c -> b n_query c', b=B)[:, [i]],
                    B,
                    self.bongard_args.n_way,
                    self.bongard_args.n_shot,
                    1,
                )
                logits.append(logit)
            logits = torch.cat(logits, dim=1)
        else:
            logits = self.fsl_head(
                rearrange(
                    shot_feat,
                    '(b n_way n_shot) c -> b (n_way n_shot) c',
                    b=B,
                    n_shot=self.bongard_args.n_shot,
                    n_way=self.bongard_args.n_way),
                rearrange(query_feat, '(b n_query) c -> b n_query c', b=B),
                B,
                self.bongard_args.n_way,
                self.bongard_args.n_shot,
                self.bongard_args.n_query,
            )

        # caption head
        # TODO: captions for neg shot
        if self.caption_head_args and self.caption_head_args.use_unpooled:
            all_shot_embs = rearrange(
                original_shot_feat,
                '(b n_way n_shot) c T -> b n_way n_shot T c',
                b=B,
                n_shot=self.bongard_args.n_shot,
                n_way=self.bongard_args.n_way
            )
            T = all_shot_embs.shape[-2]
            pos_shot_embs = all_shot_embs[:, 0].reshape(
                B*self.bongard_args.n_shot, T, -1)
            neg_shot_embs = all_shot_embs[:, 1].reshape(
                B*self.bongard_args.n_shot, T, -1)
        else:
            all_shot_embs = rearrange(
                shot_feat,
                '(b n_way n_shot) c -> b n_way n_shot c',
                b=B,
                n_shot=self.bongard_args.n_shot,
                n_way=self.bongard_args.n_way
            )
            pos_shot_embs = all_shot_embs[:, 0].reshape(
                B*self.bongard_args.n_shot, -1)
            neg_shot_embs = all_shot_embs[:, 1].reshape(
                B*self.bongard_args.n_shot, -1)

        if self.caption_head:
            captions = []
            for caption in batch['caption']:
                for _ in range(self.bongard_args.n_shot):
                    captions.append(caption)
            loss_caption = self.caption_head(
                pos_shot_embs, captions
            )
        else:
            loss_caption = 0

        if self.caption_head and generate_text:
            gen_captions = self.caption_head.generate(
                torch.cat([pos_shot_embs, neg_shot_embs], dim=0)
            )
        else:
            gen_captions = []

        return logits, loss_caption, gen_captions

    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()
        del batch_idx
        logits, loss_caption, _ = self._pred(batch)
        logits = logits.view(-1, self.bongard_args.n_way)
        loss_bongard = F.cross_entropy(logits, batch['query_labels'].view(-1))
        self.log('loss/train_bongard_loss', loss_bongard, sync_dist=True)
        self.log('loss/train_caption_loss', loss_caption, sync_dist=True)

        num_concept = [0 for _ in range(logits.size(0))]
        is_commonsense = [False for _ in range(logits.size(0))]
        tmp1 = [len(c.split()) for c in batch['concept']]
        tmp2 = [True if c != '0' else False for c in batch['commonSense']]
        for i in range(self.bongard_args.n_query):
            num_concept[i::self.bongard_args.n_query] = tmp1
            is_commonsense[i::self.bongard_args.n_query] = tmp2
        mat = (torch.argmax(logits, dim=1) ==
               batch['query_labels'].view(-1)).float()

        acc = mat.mean()
        self.train_acc.update(acc.item(), logits.size(0))
        ind = list(
            filter(lambda x: num_concept[x] == 2, list(range(logits.size(0)))))
        self.train_acc_concept_2.update(mat[ind].mean(), len(ind))
        ind = list(
            filter(lambda x: num_concept[x] == 3, list(range(logits.size(0)))))
        self.train_acc_concept_3.update(mat[ind].mean(), len(ind))
        ind = list(
            filter(lambda x: num_concept[x] == 4, list(range(logits.size(0)))))
        self.train_acc_concept_4.update(mat[ind].mean(), len(ind))
        ind = list(
            filter(lambda x: num_concept[x] == 5, list(range(logits.size(0)))))
        self.train_acc_concept_5.update(mat[ind].mean(), len(ind))
        ind = list(
            filter(lambda x: is_commonsense[x], list(range(logits.size(0)))))
        self.train_acc_commonsense.update(mat[ind].mean(), len(ind))
        ind = list(
            filter(lambda x: not is_commonsense[x], list(range(logits.size(0)))))
        self.train_acc_non_commonsense.update(mat[ind].mean(), len(ind))

        opt1.zero_grad()
        opt2.zero_grad()
        self.manual_backward(loss_bongard + self.caption_loss_coeff * loss_caption)
        opt1.step()
        opt2.step()

        sch1, sch2 = self.lr_schedulers()
        sch1.step()
        sch2.step()

        return loss_bongard + self.caption_loss_coeff * loss_caption

    def validation_step(self, batch, batch_idx):
        del batch_idx
        logits, loss_caption, gen_captions = self._pred(
            batch, generate_text=True)
        images = rearrange(batch['shot_ims'], 'b n ... -> (b n) ...')
        if gen_captions and random.random() > 0.8:
            ind = random.randint(a=0, b=len(gen_captions)-1)
            img = images[ind].permute(1, 2, 0).detach().cpu().numpy()
            caption = gen_captions[ind]
            self.viz_caption_buffer.append((img, caption))
            if len(self.viz_caption_buffer) > self.max_viz_caption:
                self.viz_caption_buffer.pop(0)

        logits = logits.view(-1, self.bongard_args.n_way)
        loss_bongard = F.cross_entropy(logits, batch['query_labels'].view(-1))
        self.log('loss/val_bongard_loss', loss_bongard, sync_dist=True)
        self.log('loss/val_caption_loss', loss_caption, sync_dist=True)

        num_concept = [0 for _ in range(logits.size(0))]
        is_commonsense = [False for _ in range(logits.size(0))]
        tmp1 = [len(c.split()) for c in batch['concept']]
        tmp2 = [True if c != '0' else False for c in batch['commonSense']]
        for i in range(self.bongard_args.n_query):
            num_concept[i::self.bongard_args.n_query] = tmp1
            is_commonsense[i::self.bongard_args.n_query] = tmp2
        mat = (torch.argmax(logits, dim=1) ==
               batch['query_labels'].view(-1)).float()

        acc = mat.mean()
        self.val_acc.update(acc.item(), logits.size(0))
        ind = list(
            filter(lambda x: num_concept[x] == 2, list(range(logits.size(0)))))
        self.val_acc_concept_2.update(mat[ind].mean(), len(ind))
        ind = list(
            filter(lambda x: num_concept[x] == 3, list(range(logits.size(0)))))
        self.val_acc_concept_3.update(mat[ind].mean(), len(ind))
        ind = list(
            filter(lambda x: num_concept[x] == 4, list(range(logits.size(0)))))
        self.val_acc_concept_4.update(mat[ind].mean(), len(ind))
        ind = list(
            filter(lambda x: num_concept[x] == 5, list(range(logits.size(0)))))
        self.val_acc_concept_5.update(mat[ind].mean(), len(ind))
        ind = list(
            filter(lambda x: is_commonsense[x], list(range(logits.size(0)))))
        self.val_acc_commonsense.update(mat[ind].mean(), len(ind))
        ind = list(
            filter(lambda x: not is_commonsense[x], list(range(logits.size(0)))))
        self.val_acc_non_commonsense.update(mat[ind].mean(), len(ind))

    def test_step(self, batch, batch_idx):
        del batch_idx
        logits, loss_caption, _ = self._pred(batch)
        logits = logits.view(-1, self.bongard_args.n_way)
        loss_bongard = F.cross_entropy(logits, batch['query_labels'].view(-1))
        self.log('loss/test_bongard_loss', loss_bongard, sync_dist=True)
        self.log('loss/test_caption_loss', loss_caption, sync_dist=True)

        num_concept = [0 for _ in range(logits.size(0))]
        is_commonsense = [False for _ in range(logits.size(0))]
        problem_id = ['' for _ in range(logits.size(0))]
        tmp1 = [len(c.split()) for c in batch['concept']]
        tmp2 = [True if c != '0' else False for c in batch['commonSense']]
        for i in range(self.bongard_args.n_query):
            num_concept[i::self.bongard_args.n_query] = tmp1
            is_commonsense[i::self.bongard_args.n_query] = tmp2
            problem_id[i::self.bongard_args.n_query] = [
                c+f'_q{i}' for c in batch['uid']]
        mat = (torch.argmax(logits, dim=1) ==
               batch['query_labels'].view(-1)).float()

        # record wrong answer
        for m, ind in zip(mat, problem_id):
            if not m:
                self.wrong_q.append(ind)

        acc = mat.mean()
        self.test_acc.update(acc.item(), logits.size(0))
        ind = list(
            filter(lambda x: num_concept[x] == 2, list(range(logits.size(0)))))
        self.test_acc_concept_2.update(mat[ind].mean(), len(ind))
        ind = list(
            filter(lambda x: num_concept[x] == 3, list(range(logits.size(0)))))
        self.test_acc_concept_3.update(mat[ind].mean(), len(ind))
        ind = list(
            filter(lambda x: num_concept[x] == 4, list(range(logits.size(0)))))
        self.test_acc_concept_4.update(mat[ind].mean(), len(ind))
        ind = list(
            filter(lambda x: num_concept[x] == 5, list(range(logits.size(0)))))
        self.test_acc_concept_5.update(mat[ind].mean(), len(ind))
        ind = list(
            filter(lambda x: is_commonsense[x], list(range(logits.size(0)))))
        self.test_acc_commonsense.update(mat[ind].mean(), len(ind))
        ind = list(
            filter(lambda x: not is_commonsense[x], list(range(logits.size(0)))))
        self.test_acc_non_commonsense.update(mat[ind].mean(), len(ind))
        self.test_acc_pos.update(mat[::2].mean(), mat.size(0))
        self.test_acc_neg.update(mat[1::2].mean(), mat.size(0))

    def all_params(self):
        base_params = []
        if self.image_encoder and not self.freeze_image_encoder:
            base_params.extend(list(self.image_encoder.parameters()))
        if self.relational_encoder:
            base_params.extend(list(self.relational_encoder.parameters()))
        if self.fsl_head:
            base_params.extend(list(self.fsl_head.parameters()))
        if self.caption_head:
            if self.caption_head_args.freeze_lm:
                base_params.extend(list(self.caption_head.proj.parameters()))
                if self.caption_head.soft_prompt is not None:
                    base_params.append(self.caption_head.soft_prompt)
            else:
                base_params.extend(list(self.caption_head.parameters()))

        return base_params

    def all_params_exclude_image_encoder(self):
        base_params = []
        if self.relational_encoder:
            base_params.extend(list(self.relational_encoder.parameters()))
        if self.fsl_head:
            base_params.extend(list(self.fsl_head.parameters()))
        if self.caption_head:
            if self.caption_head_args.freeze_lm:
                base_params.extend(list(self.caption_head.proj.parameters()))
                if self.caption_head.soft_prompt is not None:
                    base_params.append(self.caption_head.soft_prompt)
            else:
                base_params.extend(list(self.caption_head.parameters()))

        return base_params

    def image_encoder_params(self):
        base_params = []
        base_params = []
        if self.image_encoder and not self.freeze_image_encoder:
            base_params.extend(list(self.image_encoder.parameters()))

        return base_params

    def configure_optimizers(self):
        optimizer1, lr_scheduler1, update_scheduler_interval1 = make_optimizer(
            self.all_params_exclude_image_encoder(),
            self.optimizer_name,
            **self.optimizer_args
        )
        self.optimizer_args['lr'] = self.additional_args.image_encoder_lr
        optimizer2, lr_scheduler2, update_scheduler_interval2 = make_optimizer(
            self.image_encoder_params(),
            self.optimizer_name,
            **self.optimizer_args
        )
        return (
            {
                'optimizer': optimizer1,
                'lr_scheduler': {
                    'scheduler': lr_scheduler1,
                    'interval': update_scheduler_interval1,
                    'frequency': 1,
                }
            },
            {
                'optimizer': optimizer2,
                'lr_scheduler': {
                    'scheduler': lr_scheduler2,
                    'interval': update_scheduler_interval2,
                    'frequency': 1,
                }
            },
        )
