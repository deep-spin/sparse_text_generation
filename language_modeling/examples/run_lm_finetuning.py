# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from entmax import SparsemaxLoss, Entmax15Loss, EntmaxBisectLoss, sparsemax, entmax15, entmax_bisect

from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForMaskedLM, BertTokenizer,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,PreTrainedTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)

from scipy.stats import entropy

import multiprocessing
from itertools import zip_longest
from collections import defaultdict, Counter
from nltk.util import ngrams
import re
RETOK = re.compile(r'\w+|[^\w\s]|\n', re.UNICODE)


def compute_jsd(p, q, base=np.e):
    p, q = np.asarray(p.cpu()), np.asarray(q.cpu())
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    ent = entropy(p,m, base=base)/2. + entropy(q, m, base=base)/2.
    if ent==float('Inf'):
        ent=torch.log(torch.FloatTensor([2]))
    return ent

def compute_sp(p, target):
    p=np.asarray(p.cpu())
    return 1-(0.5*np.linalg.norm(p)**2 - p[target]+0.5)


def softmax_temperature(X, temperature = 1.0, axis = None):
    X=X.squeeze(0)
    for i in range(len(X)):
        X[i] = X[i]*(1/temperature)
    p = torch.softmax(X, dim=-1)
    return p

logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer)
}
def process_chunk(d):
    """Replace this with your own function
    that processes data one line at a
    time"""

    d = d.strip() + ' processed'
    return d 

def grouper(n, iterable, padvalue=None):
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)


        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def load_and_cache_examples(args, tokenizer: PreTrainedTokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    dataset = TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def mask_tokens(inputs, tokenizer: PreTrainedTokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf'), gen_func=torch.softmax):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    #assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    if top_k > 0:
        for i in range(logits.size(0)):
            indices_to_remove = logits[i] < torch.topk(logits[i], top_k)[0][..., -1, None]
            logits[i][indices_to_remove] = filter_value

    for i in range(logits.size(0)):
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits[i], descending=True)
            cumulative_probs = torch.cumsum(gen_func(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[i][indices_to_remove] = filter_value
    return logits


def sample_sequence(model, prefix_batch, prefix_length, continuation_length, top_k, top_p):
    continuation_logits = []
    context = prefix_batch
    assert context.size(1) == prefix_length

    prev = context
    output = context
    past = None
    for i in range(continuation_length):
        logits, past = model(prev, past=past)

        logits = logits[:, -1, :]
        
        prev = logits.argmax(dim=1, keepdim=True)
        
        continuation_logits.append(logits)
        output = torch.cat((output, prev), dim=1)

    continuation_logits = torch.stack(continuation_logits, 1)
    return output, continuation_logits

def ul_seq(model, batch, args):
    input_sequence = batch.cuda()
    batch = model.batch_input_sequence_by_prefix_length(input_sequence,50)
    completions, continuation_logits = sample_sequence(model, batch, 50, 100, args.top_k, args.top_p)
    pred_toks = completions[:, 50:].contiguous()
    mask = ngram_repeat_mask(pred_toks, 4).type_as(continuation_logits)
    lprobs = torch.nn.functional.log_softmax(continuation_logits, dim=-1)
    pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(1, pred_toks.view(-1, 1))
    one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=1e-20).view(pred_toks.size(0), pred_toks.size(1))
    loss = -torch.log(one_minus_probs) * mask
    loss = loss.sum()
    ntokens = pred_toks.numel()  # number of output tokens (tokens in completions)

    loss = loss / ntokens
    return loss


def ngram_repeat_mask(xs, n):
    mask = torch.zeros_like(xs)
    for i, x in enumerate(xs):
        seen = set()
        xl = x.tolist()
        for j in range(len(x)-n):
            ng = tuple(xl[j:j+n])
            if ng in seen:
                mask[i, j:j+n] = 1
            seen.add(ng)
    return mask

    
def repeat_at_1(args, predictions, targets, context_length, topk=0, topp=0.0):
    with torch.no_grad():

        predictions = torch.tensor(predictions).cuda()
        targets=targets.unsqueeze(0)
        T = targets.size(1)
        assert predictions.size(0) == T

        # T x T where prev_targets[t, :] = [y_1,...,y_t-1, -1, -1,..., -1]
        prev_targets = targets.expand(T, T).tril().masked_fill_(torch.ones_like(targets.expand(T, T)).byte().triu().bool(), -1)

        # each row t is [-1, ..., -1, y_{t-k-1}, ..., y_{t-1}, -1, ..., -1] where k is context length
        prev_targets = prev_targets.masked_fill_(torch.ones_like(targets.expand(T, T)).byte().tril(-(context_length+1)).bool(), -1)

        repeat_at_1 = (predictions[:, None] == prev_targets)
        has_repeat_at_1 = repeat_at_1.sum(1).gt(0)
        total_repeat_at_1 = has_repeat_at_1.sum()

        is_incorrect = (predictions != targets.view(-1)).view(-1, 1)
        total_wrong_repeat_at_1 = ((repeat_at_1 * is_incorrect).sum(1).gt(0)).sum()

        total_human_repeat_at_1 = (prev_targets == targets.view(T, 1)).sum(1).gt(0).sum()

    return total_repeat_at_1.item()/float(targets.size(1)), total_wrong_repeat_at_1.item()/float(targets.size(1))

def train(args, train_dataset, model, tokenizer: PreTrainedTokenizer, gen_func, eval_metric='jsd'):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        if args.loss=='entmax':
            tb_writer = SummaryWriter(filename_suffix='_entmax_'+str(args.entmax_alpha))
        elif args.top_p>0:
            tb_writer = SummaryWriter(filename_suffix='_nucleus_'+str(args.top_p))
        elif args.loss=='entmax15':
            tb_writer = SummaryWriter(filename_suffix='_entmax_1.5')
        elif args.loss=='sparsemax':
            tb_writer = SummaryWriter(filename_suffix='_sparsemax')
        elif args.loss=='cross_entropy':
            tb_writer = SummaryWriter(filename_suffix='_softmax')
        elif args.loss=='entmax_alpha':
            tb_writer = SummaryWriter(filename_suffix='_entmax_alpha')

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, shuffle=False ,sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
   
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    #logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    best_jsd=100000
    best_ppl=100000
    best_sp=0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()

            if args.unlikelihood_seq and torch.rand(1).item() < 0.5:
                if inputs.size(1) < 50:
                    continue
                else:
                    loss = ul_seq(model, inputs, args)

            elif args.loss=="entmax_alpha":
                outputs, _ = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels, unlikelihood=False, unlikelihood_seq=False)
                loss = outputs[0]
            else:
                outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels, unlikelihood=False, unlikelihood_seq=False)
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, jsd, ppl, sp = evaluate(args, model, tokenizer, loss_train=loss, gen_func=gen_func, metric=eval_metric, top_p=args.top_p)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    if jsd<best_jsd:
                        best_jsd=jsd
                        output_dir = os.path.join(args.output_dir+'/best_jsd', 'checkpoint')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)
                    if ppl<best_ppl:
                        best_ppl=ppl
                        output_dir = os.path.join(args.output_dir+'/best_ppl', 'checkpoint')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)
                    if sp>best_sp:
                        best_sp=sp
                        output_dir = os.path.join(args.output_dir+'/best_sp', 'checkpoint')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                        logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer: PreTrainedTokenizer, loss_train=0, prefix="", gen_func=torch.softmax, metric='jsd', top_p=0):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    perp = 0.0
    nb_eval_steps = 0
    model.eval()


    jsd=0
    sp=0
    repeat_16=[]
    wrong_repeat_16=[]
    repeat_32=[]
    wrong_repeat_32=[]
    repeat_128=[]
    wrong_repeat_128=[]
    repeat_512=[]
    wrong_repeat_512=[]

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = batch.to(args.device)

        with torch.no_grad():
            outputs = model(batch, masked_lm_labels=batch) if args.mlm else model(batch, labels=batch, unlikelihood=False, unlikelihood_seq=False)


            shift_logits = outputs[1][..., :-1, :].contiguous()
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = batch[..., 1:].contiguous().squeeze(0)
            
            if args.temp!=0:
                probs=softmax_temperature(shift_logits,temperature=args.temp, axis=1)
            elif args.top_p>0 or args.top_k>0:
                shift_logits = top_k_top_p_filtering(shift_logits, top_p=args.top_p, top_k=args.top_k, gen_func=gen_func)
                probs=gen_func(shift_logits,dim=1)
            else:
                probs=gen_func(shift_logits,dim=1)
            lprobs=probs

            if  len(probs[0].nonzero())!=len(probs[0]):
                probs = probs[:,:]+args.epsilon
                sums = [probs[i].sum().item() for i in range(probs.size(0))]
                probs = [probs[i]/sums[i]  for i in range(len(sums))]
            
                probs=torch.stack(probs)

            p = [probs[i,shift_labels.squeeze(0)[i].item()] for i in range(len(shift_labels.squeeze(0)))]
            p=torch.stack(p)
            perp+= torch.log(p**-1).mean().item()

            jsd_batch=[]
            labels=torch.zeros(len(shift_labels),shift_logits.size(-1))
            for i in range(len(shift_labels)):
                labels[i,shift_labels[i]] = 1
                jsd_ = compute_jsd(lprobs[i], labels[i])
                jsd_batch.append(jsd_)
            
            jsd_batch = torch.tensor(jsd_batch).mean()
            jsd+=jsd_batch

            sp_batch=[]
            for i in range(len(shift_labels)):
                sp_batch.append(compute_sp(lprobs.squeeze(0)[i], shift_labels[i]).item())

            sp_batch = torch.tensor(sp_batch).mean()
            sp+=sp_batch          


            pred = torch.multinomial(lprobs, num_samples=1).squeeze(1).view(-1).tolist()

            repeat, wrong_repeat = repeat_at_1(args, pred, shift_labels, 16, topk=args.top_k, topp=top_p)
            repeat_16.append(repeat)
            wrong_repeat_16.append(wrong_repeat)
            repeat, wrong_repeat = repeat_at_1(args, pred, shift_labels, 32, topk=args.top_k, topp=top_p)
            repeat_32.append(repeat)
            wrong_repeat_32.append(wrong_repeat)
            repeat, wrong_repeat = repeat_at_1(args, pred, shift_labels, 128, topk=args.top_k, topp=top_p)
            repeat_128.append(repeat)
            wrong_repeat_128.append(wrong_repeat)
            repeat, wrong_repeat = repeat_at_1(args, pred, shift_labels, 512, topk=args.top_k, topp=top_p)
            repeat_512.append(repeat)
            wrong_repeat_512.append(wrong_repeat)


    a = perp / len(eval_dataloader)
    perplexity = torch.exp(torch.tensor(a))
        
    jsd=jsd/len(eval_dataloader)
    sp=sp/len(eval_dataloader)


    result = {
        "sp": sp,
        "JSD": jsd,
        "perplexity" :perplexity,
        "Loss": loss_train}

    print('perplexity:', perplexity)
    print('js:', jsd)
    print('sp;', sp)
    print('repeat_16:', np.array(repeat_16).mean())
    print('wrong_repeat_16:', np.array(wrong_repeat_16).mean())
    print('repeat_32:', np.array(repeat_32).mean())
    print('wrong_repeat_32:', np.array(wrong_repeat_32).mean())
    print('repeat_128:', np.array(repeat_128).mean())
    print('wrong_repeat_128:', np.array(wrong_repeat_128).mean())
    print('repeat_512:', np.array(repeat_512).mean())
    print('wrong_repeat_512:', np.array(wrong_repeat_512).mean())
    output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    with open(output_eval_file, "a") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result, jsd, perplexity, sp


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="gpt2", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--mode", default="finetune", type=str)
    parser.add_argument("--epsilon", default=0.000001, type=float)

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=6.25e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--loss", default="cross_entropy", type=str,
                        help="Loss function to use for fine-tuning (only for GPT-2 so far)")
    parser.add_argument("--top_p", type=float, default=0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--entmax_alpha", type=float, default=1.5)
    parser.add_argument("--entmax_k", type=int, default=512)
    parser.add_argument("--entmax_bisect_iter", type=int, default=50)
    parser.add_argument("--eval_metric", type=str, default = 'jsd')
    parser.add_argument('--logging_steps', type=int, default=10000,
                        help="Log every X updates steps.")
    parser.add_argument("--temp", type=float, default=0)
    parser.add_argument('--save_steps', type=int, default=10000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    parser.add_argument('--unlikelihood_seq', action='store_true')
    args = parser.parse_args()

    if args.model_type in ["bert", "roberta"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")
    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if args.mode=='finetune':
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    else:
        config = config_class()

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len  # Our input block size will be the max possible for the model

    
    generic_entmax_loss = partial(
        EntmaxBisectLoss,
        alpha=args.entmax_alpha,
        n_iter=args.entmax_bisect_iter
    )

    
    loss_funcs = {"cross_entropy": nn.CrossEntropyLoss,
                  "sparsemax": partial(SparsemaxLoss, k=args.entmax_k),
                  "entmax15": partial(Entmax15Loss, k=args.entmax_k),
                  "entmax": generic_entmax_loss,
                  "entmax_alpha": "entmax_alpha"}

    assert args.loss in loss_funcs
    loss_func = loss_funcs[args.loss]


    generic_entmax = partial(
        entmax_bisect,
        alpha=args.entmax_alpha,
        n_iter=args.entmax_bisect_iter
    )

    gen_funcs = {"softmax": torch.softmax,
                 "sparsemax": partial(sparsemax, k=args.entmax_k),
                 "entmax15": partial(entmax15, k=args.entmax_k),
                 "entmax": generic_entmax,
                 "entmax_alpha": "entmax_alpha"}

    if args.loss=="cross_entropy":
        gen_func = gen_funcs["softmax"]
    elif args.loss=="sparsemax":
        gen_func = gen_funcs["sparsemax"]
    elif args.loss=="entmax15":
        gen_func = gen_funcs["entmax15"]
    elif args.loss=="entmax":
        gen_func = gen_funcs["entmax"]
    elif args.loss=="entmax_alpha":
        gen_func = gen_funcs["entmax_alpha"]

    if args.mode=='finetune':
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf='.ckpt' in args.model_name_or_path,
            config=config,
            loss=loss_func,
            gen_func=gen_func,
            mode=args.mode
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config, loss=loss_func, gen_func=gen_func)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, gen_func)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir, loss=loss_func, gen_func=gen_func)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint, loss=loss_func, gen_func=gen_func)
            model.to(args.device)
            result, jsd, ppl, sp = evaluate(args, model, tokenizer, prefix=global_step, gen_func=gen_func, metric=args.eval_metric, top_p=args.top_p)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
