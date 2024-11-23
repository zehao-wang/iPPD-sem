import torch
import torch.nn.functional as F
import warnings
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
import os
from models.extractor_local_rep import build_extractor
from transformers import  BertModel, BertConfig
# from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertOnlyMLMHead # transformers > 4
from transformers.modeling_bert import BertEncoder, BertPooler, BertOnlyMLMHead # transformers < 4
from torch.nn.utils.rnn import pad_sequence 
from statistics import mean
import itertools
import json
import numpy as np
from metrics import MLN_SuccessRate, MLN_Test_Metric
from utils.pos_emb import get_embedder
from utils.parse_wordmap import load_embeddings
import time
import jsonlines

# NOTE: official solution for logging on console, file handler is declared in run.py
import logging
logger = logging.getLogger("pytorch_lightning.core")

class ModifiedBert(BertModel):
    def __init__(self, config):
        super().__init__(config)
        self.mlm_head = BertOnlyMLMHead(config)
        self.mask_rate = 0.1
        self.embedding_layer = load_embeddings()
        self.instruction_proj = nn.Linear(50,768)

    def forward(
        self,
        input_ids,
        path_obj_emb,
        attention_mask,
        token_type_ids,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,

    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        self.instruction_len = input_ids.shape[1]
        if self.training:
            # MLM aux task
            # create random array of floats with equal dimensions to input_ids tensor
            self.labels = input_ids.clone()
            rand = torch.rand(input_ids.shape).to(input_ids.device)
            # create mask array
            mask_arr = (rand < self.mask_rate ) & (input_ids != 1908) & \
                       (input_ids != 0) # use word 'separate' as start token: 1908
            selection = []

            for i in range(input_ids.shape[0]):
                selection.append(
                    torch.nonzero(mask_arr[i]).flatten().tolist()
                )
            for i in range(input_ids.shape[0]):
                input_ids[i, selection[i]] = 1353 # glove mask is 1353
                mask = torch.ones(self.instruction_len, dtype=bool)
                mask[selection[i]] = False
                self.labels[i, mask] = -100
            inputs_embeds = self.embedding_layer(input_ids)   
        else:
            inputs_embeds = self.embedding_layer(input_ids)

        inputs_embeds = self.instruction_proj(inputs_embeds)

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        # NOTE: construct language + map input
        if path_obj_emb is None:
            inputs_embeds = inputs_embeds
        else:
            inputs_embeds = torch.cat([
                inputs_embeds, path_obj_emb
                ], dim=1
            )

        input_shape = inputs_embeds.size()[:-1]
        input_ids = None
    
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if self.training:
            token_pred = self.mlm_head(sequence_output[:,:self.instruction_len]).view(-1, self.config.vocab_size)
            mlm_loss = F.cross_entropy(token_pred, self.labels.view(-1))
        else:
            mlm_loss = 0
        outputs = (sequence_output, pooled_output, mlm_loss, ) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)

        # TODO: should consider gaussian smoothing further
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class PointNet_Transformer_Glove(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.obj_rot = config.obj_rot
        self.obj_3dpos = config.obj_3dpos
        if self.obj_rot:
            ptnet_input_dim = config.SEM_LABEL_DIM + (3 * 10 * 2 + 3) + (3 * 4 * 2 + 3)
        else:
            ptnet_input_dim = config.SEM_LABEL_DIM + (3 * 10 * 2 + 3)
        
        if self.obj_3dpos:
            ptnet_input_dim -= (3 * 10 * 2)

        self.obj_encoder = build_extractor(config.OBJ_ENCODER, ptnet_input_dim)
        self.agent_pose_dim =  (3 * 10 * 2 + 3) + (3 * 4 * 2 + 3) # pos emb dim + rot emb dim
        self.path_feat_proj = nn.Linear(self.agent_pose_dim + config.JOINT_ENCODER.hidden_size, config.JOINT_ENCODER.hidden_size)

        self.join_model_config = BertConfig.from_pretrained('bert-base-uncased')
        if config.JOINT_ENCODER.pretrained_emb:
            self.joint_model = ModifiedBert.from_pretrained("bert-base-uncased")
        else:
            # NOTE: add reinit for bert
            self.joint_model = ModifiedBert(self.join_model_config)

        self.join_model_config.hidden_size = config.JOINT_ENCODER.hidden_size
        self.join_model_config.num_hidden_layers= config.JOINT_ENCODER.num_layers

        self.joint_model.encoder = BertEncoder(self.join_model_config)
        self.joint_model.pooler = BertPooler(self.join_model_config)
        self.joint_model.embeddings.token_type_embeddings = nn.Embedding(4, config.OBJ_ENCODER.hidden_size)

        self.reg_head1 = nn.Sequential( # main target
            nn.Linear(config.OBJ_ENCODER.hidden_size, 1), # 1 score output
            # nn.Sigmoid()
        )
        self.reg_head2 = nn.Sequential( # main target
            nn.Linear(config.OBJ_ENCODER.hidden_size, 1), # 1 score output
            # nn.Sigmoid()
        )

        self.loss_fn = nn.MSELoss()
        
        self.mln_success_rate = MLN_SuccessRate()
        self.mln_test_container = MLN_Test_Metric()

        self.ablate_type =config.ablate_type
        # self.sem_label_embedding = load_sem_embeddings() # nn.Embedding(50, config.SEM_LABEL_DIM)
        self.sem_label_embedding = self.joint_model.embedding_layer # nn.Embedding(50, config.SEM_LABEL_DIM)

        self.discrete_dir = config.discrete_dir
        if self.discrete_dir:
            self.agent_rot_emb = nn.Embedding(15, (3 * 4 * 2 + 3))
        else:
            self.agent_rot_emb, _ = get_embedder(4)
        
        self.pos_emb_linear = nn.Linear(90, config.OBJ_ENCODER.hidden_size)
        self.pose_cat = config.POSE_CAT

    def forward(self, batch):
        # inference actions
        instruction_tokens, (agent_pos_embs, agent_rots), (obs_pos_embs, obs_rots, sem_labels), _, _, _, seq_lens = batch

        # ============= concate imp ============
        sems_emb = self.sem_label_embedding(sem_labels)
        # sems_emb = sem_labels
        path_feats = self.obj_encoder(torch.cat([obs_pos_embs[:,:,:3], sems_emb], dim=2).float())
        path_obj_emb = pad_sequence(torch.split(path_feats, seq_lens), batch_first=True) 
        # ====================================

        # Step 1: encode paths' feature
        # path_feats = []
        # # for (obs_pos_emb, obs_rot, sems) in zip(obs_pos_embs, obs_rots, sem_labels):
        # for (obs_pos_emb, sems) in zip(obs_pos_embs, sem_labels):
        #     sems_emb = self.sem_label_embedding(sems)
        #     # obs_rot_emb = self.agent_rot_emb(obs_rot)
        #     if self.obj_3dpos:
        #         obs_pos_emb = obs_pos_emb[:, :, :3]
                
        #     if self.obj_rot:
        #         #obs_emb = self.obj_encoder(torch.cat([obs_pos_emb, obs_rot_emb, sems_emb], dim=2).float())
        #         raise NotImplementedError()
        #     else:
        #         obs_emb = self.obj_encoder(torch.cat([obs_pos_emb, sems_emb], dim=2).float())
        #     # ==> [num_obs, hidden_size]
        #     path_feats.append(obs_emb) 
        # path_obj_emb = pad_sequence(path_feats, batch_first=True)
        
        # ===========================================================

        agent_feats = []
        for agent_pos_emb, agent_rot in zip(agent_pos_embs, agent_rots):
            agent_rot_emb = self.agent_rot_emb(agent_rot)
            agent_feat = torch.cat([agent_pos_emb, agent_rot_emb], dim=1).float()
            agent_feats.append(agent_feat)
        
        agent_feats = pad_sequence(agent_feats, batch_first=True).float()
        
        if self.ablate_type == 'agent_pos':
            agent_feats.fill_(0.)

        if self.ablate_type == 'compass':
            path_obj_emb.fill_(0.)

        if self.pose_cat:
            # NOTE: concate version
            path_emb = torch.cat([agent_feats, path_obj_emb], dim=2) 
            path_emb = self.path_feat_proj(path_emb)
        else:
            # NOTE: addition version
            # import ipdb;ipdb.set_trace() # breakpoint 275
            agent_feats = self.pos_emb_linear(agent_feats)
            path_emb = agent_feats + path_obj_emb
        
        obj_type_ids = []
        map_attn_mask = []
        for seq_len in seq_lens:
            # token type 0 for empty, 1 for instruction, 3 for path feats
            obj_type_ids.append( torch.ones(seq_len, )*3)
            map_attn_mask.append(torch.ones(seq_len, ))
        obj_type_ids = pad_sequence(obj_type_ids, batch_first=True)
        map_attn_mask = pad_sequence(map_attn_mask, batch_first=True)
        
        # Step 2: preprocess text input
        # instruction_emb = self.glove_linear(instructions) # For glove instruction [bs, 200, hidden_size]
        if self.ablate_type == 'text':
            instruction_tokens['input_ids'] = instruction_tokens['input_ids'][:, 0:1]
            instruction_tokens['token_type_ids'] = instruction_tokens['token_type_ids'][:, 0:1]
            instruction_tokens['attention_mask'] = instruction_tokens['attention_mask'][:, 0:1]

        if self.ablate_type == 'text_only':
            out = self.joint_model(**instruction_tokens, path_obj_emb=None)
        else:
            instruction_tokens['token_type_ids'] = torch.cat([
                instruction_tokens['token_type_ids'], obj_type_ids.to(path_emb.device).long()
            ], dim=1)
            instruction_tokens['attention_mask'] = torch.cat([
                instruction_tokens['attention_mask'], map_attn_mask.to(path_emb.device).long()
            ], dim=1)

            out = self.joint_model(**instruction_tokens, path_obj_emb=path_emb)
    
        pred_tar = self.reg_head1(out[0][:, 0]).squeeze(1)
        pred_ndtw = self.reg_head2(out[0][:, 0]).squeeze(1)
        mlm_loss = out[2]
        return pred_tar, pred_ndtw, mlm_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        # training_step defined the train loop.
        _, _, _, target_ndtw, target_score, infos, seq_len = train_batch

        pred_tar, pred_ndtw, mlm_loss = self(train_batch)
        loss_tar = self.loss_fn(pred_tar, target_score)
        loss_ndtw = self.loss_fn(pred_ndtw, target_ndtw)
        loss = loss_tar  + loss_ndtw
        # loss_tar=torch.tensor(0);loss_ndtw=torch.tensor(0)
        # loss = self.loss_fn(pred_tar, target_score * 0.5 + target_ndtw * 0.5)

        loss += mlm_loss
        self.log('L-tar', loss_tar, batch_size=target_score.shape[0], prog_bar=True)
        self.log('L-ndtw', loss_ndtw, batch_size=target_ndtw.shape[0], prog_bar=True)
        self.log('L-mlm', mlm_loss, batch_size=target_score.shape[0], prog_bar=True)
        
        return loss

    def validation_step(self, val_batch, batch_idx):
        # validation_step defined the validation loop.
        _, _, _, target_ndtw, target_score, infos, seq_len = val_batch

        pred_tar, pred_ndtw, _ = self(val_batch)
        loss_tar = self.loss_fn(pred_tar, target_score)
        loss_ndtw = self.loss_fn(pred_ndtw, target_ndtw)
        # loss_tar=torch.tensor(0);loss_ndtw=torch.tensor(0)

        self.log('val_L-tar', loss_tar, batch_size=target_score.shape[0])
        self.log('val_L-ndtw', loss_ndtw, batch_size=target_ndtw.shape[0])
        
        out_list = []
        preds = (pred_tar + pred_ndtw).flatten().tolist()
        # preds = pred_tar.flatten().tolist()
        for pred, info in zip(preds, infos):
            info.update({"pred": pred})
            out_list.append(info)
        
        self.mln_success_rate(out_list)
        
        return {"loss-tar": loss_tar, "loss-ndtw": loss_ndtw, "out_list": out_list}

    def validation_epoch_end(self, outputs) -> None:
        out_dict = {}

        val_set_eval_results = list(itertools.chain.from_iterable([out['out_list'] for out in outputs]))

        values, counts = np.unique([res['dis_score'] for res in val_set_eval_results], return_counts=True)
        val_score_loss = mean([out['loss-tar'].item() for out in outputs ])

        # NOTE: MLN success rate
        success_rate, selected_results = self.mln_success_rate.compute()
        self.log('success_rate', success_rate)
        # Collect report information
        sample_len = min(10, len(selected_results))
        out_dict = {
            "val_score_loss": val_score_loss,
            "val_distribution": sorted(list(zip(values, counts)), key=lambda x: x[0]),
            "success_rate": success_rate,
            # "sampled_results": json.dumps(selected_results[:sample_len], indent=4),
            "num_datum_evaluated": len(selected_results)
        }

        logger.info(f"\n")
        for k,v in out_dict.items():
            logger.info(f"{k}: {v}")
        logger.info(f"\n")

        out_path = os.path.join(self.trainer.logger.log_dir, f"best_routes-{time.time()}.jsonl")
        with jsonlines.open(out_path, mode='w') as writer:
            for res in selected_results:
                writer.write(res)

        self.mln_success_rate.reset()
        return out_dict
    

    def test_step(self, batch, batch_idx):
        # validation_step defined the validation loop.
        _, _, _, _, _, infos, _ = batch

        pred_tar, pred_ndtw, _ = self(batch)
        
        out_list = []
        preds = pred_tar.flatten().tolist()
        for pred, info in zip(preds, infos):
            info.update({"pred": pred})
            out_list.append(info)
        
        self.mln_test_container(out_list)
        
        return

    def test_epoch_end(self, outputs):
        _, predictions = self.mln_test_container.compute()
        out_path = os.path.join(self.trainer.logger.log_dir, "all_traj_pred.json")
        print(f"Output predictions for {len(predictions)} episodes in current split")
        with open(out_path, 'w') as f:
            json.dump(predictions, f)
        return 
     

