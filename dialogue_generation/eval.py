# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
from collections import defaultdict


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import build_input_from_segments, add_special_tokens_, pad_dataset, get_data_loaders
from utils import get_dataset, download_pretrained_model

from functools import partial
from entmax import SparsemaxLoss, Entmax15Loss, EntmaxBisectLoss, sparsemax, entmax15, entmax_bisect

import eval_utils
from scipy.stats import entropy

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
MODEL_INPUTS = ["input_ids", "reply", "token_type_ids"]

def compute_jsd(p, q, base=np.e):
    p, q = np.asarray(p.cpu()), np.asarray(q.cpu())
    p, q = p/p.sum(), q/q.sum()
    m = 1./2*(p + q)
    return entropy(p,m, base=base)/2. + entropy(q, m, base=base)/2.

def compute_sp(p, target):
    p=np.asarray(p.cpu())
    return 0.5*np.linalg.norm(p)**2 - p[target]+0.5

def build_input(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history
    sequence = [sequence[0]] + [[speaker1 if (len(sequence)-i) % 2 else speaker2] + s for i, s in enumerate(sequence[1:])] + [[speaker1]]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["reply"] = reply
    return instance

def softmax_temperature(X, temperature = 1.0, axis = None):
    for i in range(len(X)):
        X[i] = X[i]*(1/temperature)
    p = torch.softmax(X, dim=-1)

    return p

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    return logits



def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2', 'gpt2-medium'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=1)    
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--num_candidates", type=int, default=1)
    parser.add_argument("--personality_permutations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--distributed", action='store_true') 
    parser.add_argument("--temperature", type=int, default=1, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--entmax_alpha", type=float, default=1.5)
    parser.add_argument("--entmax_k", type=int, default=512)
    parser.add_argument("--entmax_bisect_iter", type=int, default=50)
    parser.add_argument("--loss", default="cross_entropy", type=str)
    parser.add_argument("--metric", default="jsd", type=str)
    parser.add_argument("--epsilon", default=0.000001, type=float)
    parser.add_argument("--name", default='', type=str)
    parser.add_argument("--temp", type=float, default=0)


    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    args.train_batch_size=args.batch_size
    args.valid_batch_size=args.batch_size

    generic_entmax_loss = partial(
        EntmaxBisectLoss,
        alpha=args.entmax_alpha,
        n_iter=args.entmax_bisect_iter)

    
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
        n_iter=args.entmax_bisect_iter)

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

    if args.model_checkpoint == "":
        if args.model == 'gpt2' or args.model == 'gpt2-medium':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()
	
	
    if args.seed != 0:
    	random.seed(args.seed)
    	torch.random.manual_seed(args.seed)
    	torch.cuda.manual_seed(args.seed)


    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' or args.model == 'gpt2-medium' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    bos, eos, speaker1, speaker2, pad = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    model.eval()
    if args.metric=='f1':
        datasets = {"train": defaultdict(list), "valid": defaultdict(list)}
        for dataset_name, dataset in personachat.items():
            num_candidates = 1
            if dataset_name != 'train':
                for dialog in dataset:
                    persona = dialog["personality"].copy()
                    for utterance in dialog["utterances"]:
                        history = utterance["history"]#[-(2*2+1):]
                        for j, candidate in enumerate(utterance["candidates"][-num_candidates:]):
                            instance = build_input(persona, history, candidate, tokenizer)
                            for input_name, input_array in instance.items():
                                datasets[dataset_name][input_name].append(input_array)
                   
        logger.info("Pad inputs and convert to Tensor")
        tensor_datasets = {"train": [], "valid": []}
        for dataset_name, dataset in datasets.items():
            if dataset_name != 'train':
                inputs = dataset['input_ids']
                replies = dataset['reply']
                token_type_ids = dataset['token_type_ids']                    
    
        special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        predictions=[]
        references=[]
        preds=[]
        refs=[]
        histories=[] 
        for i in range(len(inputs)):
            if i%100==0:
                print(str(i) + ' out of ' + str(len(inputs)))
            inpu = torch.tensor(inputs[i]).unsqueeze(0).cuda()
            token_ids = torch.tensor(token_type_ids[i]).unsqueeze(0).cuda()
            current_output=[]
            for i in range(args.max_length):
                if i > 0:
                    inpu = torch.cat([inpu , prev.unsqueeze(0)],1)  
                    if token_ids[0][-1]==50260:
                        token_ids = torch.cat([token_ids, torch.tensor([50260]).cuda().unsqueeze(0)],1)
                    else:
                        token_ids = torch.cat([token_ids, torch.tensor([50261]).cuda().unsqueeze(0)],1)                

                logits = model(inpu, token_type_ids=token_ids)
                if isinstance(logits, tuple):  
                    logits = logits[0]
                logits = logits[0, -1, :] 
                if args.top_k!=0 or args.top_p!=0:
                    logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
                if args.temp!=0:
                    probs=softmax_temperature(logits.unsqueeze(0),temperature=args.temp, axis=1).squeeze(0)
                else:
                    probs = gen_func(logits, dim=-1)
                
                prev = torch.multinomial(probs, 1)

                if prev.item() in special_tokens_ids:
                    break
                current_output.append(prev.item())

            out_text = tokenizer.decode(current_output, skip_special_tokens=True)
            target = tokenizer.decode(replies[i])
            history = tokenizer.decode(inputs[i])
            predictions.append(out_text)
            references.append(target)
            preds.append(current_output)
            refs.append(replies[i])

        f1_score = eval_utils.f1(preds,refs)
    
        print('F1_score:', f1_score)

        distinct_1, distinct_2, distinct_3, distinct_4 = eval_utils.distinct(predictions)

        print('distinct_1:', distinct_1)
        print('distinct_2:', distinct_2)
        print('distinct_3:', distinct_3)
        print('distinct_4:', distinct_4)

    else:
        _, val_loader, _, valid_sampler = get_data_loaders(args, tokenizer)
        jsd=0
        sp=0
        perp= 0.0
        nb_eval_steps = 0
        v=0
        for batch in val_loader:
            v+=1
            if v%100==0:
                print(str(v) + ' out of ' + str(7801))
            batch = tuple(input_tensor.to(args.device) for input_tensor in batch)
            input_ids, lm_labels, lm_labels, mc_labels, token_type_ids = batch
            lm_logits = model(input_ids, token_type_ids=token_type_ids)
            lm_logits = lm_logits[0]
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            
            lm_logits_flat_shifted = list(lm_logits_flat_shifted.cpu().detach().numpy())
            lm_labels_flat_shifted = list(lm_labels_flat_shifted.cpu().numpy())
            lm_logits_flat_shifted = [lm_logits_flat_shifted[i] if lm_labels_flat_shifted[i]!=pad and lm_labels_flat_shifted[i]!=eos else [0] for i in range(len(lm_labels_flat_shifted))]
            lm_logits_flat_shifted = torch.tensor(list(filter(lambda a:  len(a) != 1, lm_logits_flat_shifted)))
            lm_labels_flat_shifted = torch.tensor(list(filter(lambda a: a != pad and a != eos, lm_labels_flat_shifted)))
            

            if args.top_p>0 or args.top_k>0:
                j=0
                for l in lm_logits_flat_shifted:
                    j+=1
                    if j>1:
                        shift_logits = torch.cat([shift_logits, top_filtering(l, top_p=args.top_p, top_k=args.top_k).unsqueeze(0)],0)
                    else:
                        shift_logits = top_filtering(l, top_p=args.top_p, top_k=args.top_k).unsqueeze(0)
            else:
                shift_logits = lm_logits_flat_shifted
            if args.temp!=0:
                probs=softmax_temperature(shift_logits,temperature=args.temp, axis=1)
            else:
                probs=gen_func(shift_logits,dim=1)

            jsd_batch=[]
            labels=torch.zeros(len(lm_labels_flat_shifted),shift_logits.size(-1))
            for i in range(len(lm_labels_flat_shifted)):
                labels[i,lm_labels_flat_shifted[i]] = 1
                jsd_ = compute_jsd(probs[i], labels[i])
                jsd_batch.append(jsd_)
            jsd_batch = torch.tensor(jsd_batch).mean()
            jsd+=jsd_batch
            sp_batch=[]
            for i in range(len(lm_labels_flat_shifted)):
                sp_batch.append(compute_sp(probs.squeeze(0)[i], lm_labels_flat_shifted[i]))

            sp_batch = torch.tensor(sp_batch).mean()
            sp+=sp_batch

            if  len(probs[0].nonzero())!=len(probs[0]):
                probs = probs[:,:]+args.epsilon
                sums = [probs[i].sum().item() for i in range(probs.size(0))]
                probs = [probs[i]/sums[i]  for i in range(len(sums))]
                probs = torch.stack(probs)

            p = [probs[i,lm_labels_flat_shifted.squeeze(0)[i].item()] for i in range(len(lm_labels_flat_shifted.squeeze(0)))]
            p=torch.stack(p)
            perp+= torch.log(p**-1).mean().item()
                
            nb_eval_steps += 1

        jsd=jsd/nb_eval_steps
        sp=sp/nb_eval_steps
        a = perp / nb_eval_steps
        perplexity = torch.exp(torch.tensor(a))
        print('perplexity:', perplexity)
        print('jsd:', jsd)
        print('sp:', sp)



if __name__ == "__main__":
    run()
