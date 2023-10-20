import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, Blip2Model,
                          Blip2QFormerConfig, Blip2QFormerModel,
                          GenerationConfig, GPT2Config, GPT2LMHeadModel)

from models.model import register

logger = logging.getLogger(__name__)


@register('opt_captioner')
class GPT2Captioner(nn.Module):

    def __init__(self,
                 input_feat_dim,
                 lm_name='facebook/opt-125m',
                 freeze_lm=True,
                 max_length=500,
                 num_prefix_token=0,
                 image_embed_dropout_prob=0,
                 use_beam=True,
                 num_beams=4,
                 max_sample_length=32,
                 use_qformer=False,
                 pretrained_qformer=False,
                 freeze_qformer=False,
                 pretrained_query=False,
                 freeze_query=False,
                 **kwargs):
        super().__init__()
        assert freeze_lm == True
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(lm_name)

        # TBD: we will freeze the LM from a pretrained weight for now
        # self.config = GPT2Config()
        # self.lm = GPT2LMHeadModel.from_config(self.config)
        self.lm = AutoModelForCausalLM.from_pretrained(lm_name)
        self.generation_config = GenerationConfig.from_pretrained(lm_name)
        self.generation_config.do_sample = use_beam
        self.generation_config.num_beams = num_beams
        self.generation_config.max_length = max_sample_length

        self.input_embeddings = self.lm.get_input_embeddings()
        if freeze_lm:
            self.lm.eval()
            for p in self.lm.parameters():
                p.requires_grad = False

        self.use_qformer = use_qformer
        if use_qformer:
            if pretrained_qformer or pretrained_query:
                blip2 = Blip2Model.from_pretrained('Salesforce/blip2-opt-2.7b')
                del blip2.vision_model
                del blip2.language_model
                if pretrained_qformer:
                    self.qformer = blip2.qformer
                else:
                    self.qformer = Blip2QFormerModel(Blip2QFormerConfig())
                if pretrained_query:
                    self.qformer_query = blip2.query_tokens
                else:
                    # FIXME: we use 32 query tokens by default
                    self.qformer_query = nn.Parameter(torch.FloatTensor(
                        1, 32, self.qformer.config.hidden_size).uniform_(-0.5, 0.5))
            else:
                self.qformer = Blip2QFormerModel(Blip2QFormerConfig())
                # FIXME: we use 32 query tokens by default
                self.qformer_query = nn.Parameter(torch.FloatTensor(
                    1, 32, self.qformer.config.hidden_size).uniform_(-0.5, 0.5))

            self.proj = nn.Linear(input_feat_dim,
                                  self.qformer.config.encoder_hidden_size)
            assert self.qformer.config.hidden_size == self.input_embeddings.embedding_dim
        else:
            self.proj = nn.Linear(
                input_feat_dim, self.input_embeddings.embedding_dim)
        self.image_dropout = nn.Dropout(image_embed_dropout_prob)

        if freeze_qformer:
            assert pretrained_qformer
            self.qformer.eval()
            for p in self.qformer.parameters():
                p.requires_grad = False
        if freeze_query:
            assert pretrained_query
            self.qformer_query.requires_grad = False

        if num_prefix_token:
            self.soft_prompt = nn.Parameter(torch.FloatTensor(
                num_prefix_token, self.input_embeddings.embedding_dim
            ).uniform_(-0.5, 0.5))
        else:
            self.soft_prompt = None

    def forward(self, image_embs, captions):
        """
        Args:
            image_embs: (B, C)
            captions: a list of B captions
        """
        B = image_embs.size(0)
        # B, C -> B, 1, C
        image_embs = self.image_dropout(self.proj(image_embs))
        if len(image_embs.shape) != 3:
            image_embs = image_embs.unsqueeze(1)
        if self.use_qformer:
            image_embs = self.qformer(self.qformer_query.expand(B, -1, -1), encoder_hidden_states=image_embs).last_hidden_state

        input_ids = self.tokenizer(
            captions,
            return_tensors='pt',
            padding=True,
            max_length=self.max_length,
            truncation=True
        )['input_ids'].to(image_embs.device)

        # input_embs: [soft_prompt, image_emb, text_emb]
        input_embs = self.input_embeddings(input_ids)
        input_embs = torch.cat([image_embs, input_embs], dim=1)
        if self.soft_prompt is not None:
            input_embs = torch.cat(
                [self.soft_prompt.repeat(B, 1, 1), input_embs], dim=1)

        # labels: [soft_prompt(-100), image_emb(-100), text_emb, padding(-100)]
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        labels_image_emb = torch.zeros(
            image_embs.shape[:2], dtype=torch.int64).to(image_embs.device) - 100
        labels = torch.cat([labels_image_emb, labels], dim=1)
        if self.soft_prompt is not None:
            labels_soft_prompt = torch.zeros(
                B, self.soft_prompt.size(0),
                dtype=torch.int64).to(image_embs.device) - 100
            labels = torch.cat([labels_soft_prompt, labels], dim=1)

        assert labels.size(1) == input_embs.size(1)
        output = self.lm(
            inputs_embeds=input_embs,
            labels=labels,
        )
        return output.loss

    def generate(self, image_embs, return_text=True):
        """
        Args:
            image_embs: (B, C)
        """
        B = image_embs.size(0)
        # B, C -> B, 1, C
        image_embs = self.image_dropout(self.proj(image_embs))
        if len(image_embs.shape) != 3:
            image_embs = image_embs.unsqueeze(1)
        if self.use_qformer:
            image_embs = self.qformer(self.qformer_query.expand(B, -1, -1), encoder_hidden_states=image_embs).last_hidden_state
        input_embs = image_embs

        if self.soft_prompt is not None:
            input_embs = torch.cat(
                [self.soft_prompt.repeat(B, 1, 1), input_embs], dim=1)
        output = self.lm.generate(
            inputs_embeds=input_embs,
            generation_config=self.generation_config
        )
        if return_text:
            ret = []
            for o in output:
                ret.append(self.tokenizer.decode(o))
            return ret
        else:
            return output