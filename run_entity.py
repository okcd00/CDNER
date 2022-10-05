import os
import sys
import time
import json
import torch
import random
import logging
import argparse
from tqdm import tqdm
from pprint import pprint
import numpy as np

from shared.data_structures import Dataset
from shared.const import task_ner_labels, get_labelmap
from entity.utils import (
    batchify, convert_dataset_to_samples,
    logger, output_ner_predictions)
from entity.models import EntityModel
from modules.span_filter import SpanFilter
from modules.sample_augmentor import SampleAugmentor
from transformers import AdamW, get_linear_schedule_with_warmup


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
args = None
# logger = logging.getLogger('root')


def set_random_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Set {seed} as the Random Seed.")


def save_model(model, args):
    """
    Save the model to the output directory
    """
    logger.info('Saving model to %s...'%(args.output_dir))
    model_to_save = model.bert_model.module if hasattr(model.bert_model, 'module') else model.bert_model
    model_to_save.save_pretrained(args.output_dir)
    model.tokenizer.save_pretrained(args.output_dir)


def args_str2bool(_str):
    return _str.lower().strip() in ['True', 'true', '1']


def args_str2float(_str):
    return float(_str)


def fill_args(args):
    pretrained_model_dir = "/data/chendian/pretrained_bert_models/"
    if 'bert-base-chinese' in args.model:
        # default bert model directory
        args.bert_model_dir = pretrained_model_dir + 'chinese_L-12_H-768_A-12/'
    elif 'albert' in args.model:
        logger.info(f"Use Albert: {args.model}")
        args.use_albert = True
        args.bert_model_dir = pretrained_model_dir + f"{args.model}"

    if args.task == 'msra':
        args.data_dir = args.data_dir.replace('msra', 'msra_origin')
        args.output_dir = args.output_dir.replace('msra', 'msra_origin')
        
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_train:
        logger.addHandler(logging.FileHandler(
            os.path.join(args.output_dir, "train.log"), 'w'))
    else:
        logger.addHandler(logging.FileHandler(
            os.path.join(args.output_dir, "eval.log"), 'w'))

    file_name_postfix = ""
    if 'score' in args.boundary_only_mode:
        if args.take_width_feature:
            # '' for name+width feature.
            postfix_str = ''
        else:  
            # '_n' for name feature only.
            postfix_str = '_n'
        file_name_postfix = f".with_{args.boundary_only_mode}{postfix_str}"

    args.train_data = os.path.join(args.data_dir, f'train{file_name_postfix}.json')
    args.dev_data = os.path.join(args.data_dir, f'dev{file_name_postfix}.json')
    args.test_data = os.path.join(args.data_dir, f'test{file_name_postfix}.json')

    return args


def get_label_mapping(args):
    if args.boundary_only_mode in ['bin', 'bdy', 'span']:  
        # span: train a pure-model for span selection only. O(n^2) + O(n^2)
        # bin: train a pure-model for binary judgment. O(n) + O(n^2)
        # bdy: # train a pure-model for boundary token classification. O(n) + O(m^2)
        ner_label2id, ner_id2label = get_labelmap(task_ner_labels[args.boundary_only_mode])
    else:  # original labelmap
        ner_label2id, ner_id2label = get_labelmap(task_ner_labels[args.task])
    
    return ner_label2id, ner_id2label
    

def get_span_filter(args, ner_label2id, drop_with_punc=True):
    filter_method_dict = {
        'max': np.max,
        'min': np.min,
    }
    span_filter = SpanFilter(
        ner_label2id=ner_label2id,
        max_span_length=args.max_span_length,
        drop_with_punc=drop_with_punc,
        filter_method=filter_method_dict.get(
            args.boundary_det_filter_method, np.max),
        filter_threshold=args.boundary_det_threshold,
        boundary_only_mode=args.boundary_only_mode,
        method=args.span_filter_method,
    )
    return span_filter


def get_samples(data, args=None, span_filter=None, is_training=False):
    if span_filter:
        span_filter.include_positive = is_training
    _samples, _ner = convert_dataset_to_samples(
        data, 
        max_span_length=args.max_span_length if args else 25, 
        context_window=args.context_window if args else 0,
        span_filter=span_filter,
        is_training=is_training)
    return _samples, _ner


def get_optimizer(model, args, n_batches):
    # use AdamW
    param_optimizer = list(model.bert_model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer
            if 'bert' in n]},
        {'params': [p for n, p in param_optimizer
            if 'bert' not in n], 'lr': args.task_learning_rate}]
    optimizer = AdamW(optimizer_grouped_parameters, 
                      lr=args.learning_rate, correct_bias=not(args.bertadam))
    t_total = n_batches * args.num_epoch
    t_warmup = int(t_total * args.warmup_proportion)
    scheduler = get_linear_schedule_with_warmup(optimizer, t_warmup, t_total)
    return optimizer, scheduler


def evaluate(model, batches, tot_gold, output_prf=False, take_alpha_labels=False):
    """
    Evaluate the entity model
    """
    torch.cuda.empty_cache()
    logger.info('Evaluating...')
    c_time = time.time()
    cor = 0
    tot_pred = 0
    l_cor = 0
    l_tot = 0
    
    for i in range(len(batches)):
        output_dict = model.run_batch(batches[i], training=False)
        pred_ner = output_dict['pred_ner']
        for sample, preds in zip(batches[i], pred_ner):
            # print(list(zip(sample['spans_label'], preds)))
            for gold, pred in zip(sample['spans_label'], preds):
                l_tot += 1
                if pred == gold:
                    l_cor += 1
                if pred != 0 and gold != 0 and pred == gold:
                    cor += 1
                if pred != 0:
                    tot_pred += 1
                   
    acc = l_cor / l_tot
    logger.info('Accuracy: %5f'%acc)
    logger.info('Cor: %d, Pred TOT: %d, Gold TOT: %d'%(cor, tot_pred, tot_gold))
    p = cor / tot_pred if cor > 0 else 0.0
    r = cor / tot_gold if cor > 0 else 0.0
    f1 = 2 * (p * r) / (p + r) if cor > 0 else 0.0
    logger.info('P: %.5f, R: %.5f, F1: %.5f'%(p, r, f1))
    logger.info('Used time: %f'%(time.time()-c_time))
    torch.cuda.empty_cache()
    if output_prf:
        return p, r, f1
    return f1


if __name__ == '__main__':
    # parser.parse_args(['-f', 'foo', '@args.txt'])
    # in args.txt, each arg in a single line.
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')

    # paths
    parser.add_argument('--task', type=str, default=None, required=True)
    parser.add_argument('--data_dir', type=str, default=None, required=True, 
                        help="path to the preprocessed dataset")
    parser.add_argument('--output_dir', type=str, default='entity_output', 
                        help="output directory of the entity model")
    parser.add_argument('--dev_pred_filename', type=str, default="ent_pred_dev.json",
                        help="the prediction filename for the dev set")
    parser.add_argument('--test_pred_filename', type=str, default="ent_pred_test.json",
                        help="the prediction filename for the test set")

    # hyper-params
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_span_length', type=int, default=8, 
                        help="spans w/ length up to max_span_length are considered as candidates")
    parser.add_argument('--train_batch_size', type=int, default=32, 
                        help="batch size during training")
    parser.add_argument('--eval_batch_size', type=int, default=32, 
                        help="batch size during inference")
    parser.add_argument('--learning_rate', type=float, default=1e-5, 
                        help="learning rate for the BERT encoder")
    parser.add_argument('--task_learning_rate', type=float, default=1e-4, 
                        help="learning rate for task-specific parameters, i.e., classification head")
    parser.add_argument('--warmup_proportion', type=float, default=0.1, 
                        help="the ratio of the warmup steps to the total steps")
    parser.add_argument('--num_epoch', type=int, default=100, 
                        help="number of the training epochs")
    parser.add_argument('--print_loss_step', type=int, default=200, 
                        help="how often logging the loss value during training")
    parser.add_argument('--eval_per_epoch', type=int, default=1, 
                        help="how often evaluating the trained model on dev set during training")
    parser.add_argument('--context_window', type=int, default=0,
                        help="the context window size W for the entity model")

    # pre-trained model selection
    parser.add_argument('--model', type=str, default='bert-base-uncased',
                        help="the base model name (a huggingface model)")
    parser.add_argument('--use_albert', action='store_true',
                        help="whether to use ALBERT model")
    parser.add_argument('--bert_model_dir', type=str, default=None,
                        help="the base model directory")
    parser.add_argument("--bertadam", action="store_true",
                        help="If bertadam, then set correct_bias = False")

    # script switches
    parser.add_argument('--do_train', action='store_true', 
                        help="whether to run training")
    parser.add_argument('--do_eval', action='store_true', 
                        help="whether to run evaluation")
    parser.add_argument('--eval_test', action='store_true', 
                        help="whether to evaluate on test set")
    parser.add_argument('--inv_test', action='store_true',
                        help="whether to evaluate on invariance test set")

    # ablations
    parser.add_argument('--loss_function', type=str, default='ce',
                        help="which loss function to use, [ce|ls|focal]")
    parser.add_argument('--train_shuffle', action='store_true',
                        help="whether to train with randomly shuffled data")
    parser.add_argument('--take_width_feature', type=args_str2bool, default='True',
                        help="whether to take width embeddings for PURE")
    parser.add_argument('--take_name_module', type=args_str2bool, default='True',
                        help="whether to take name module for PURE")
    parser.add_argument('--take_context_module', type=args_str2bool, default='False',
                        help="whether to take context module for PURE")
    parser.add_argument('--take_context_attn', type=args_str2bool, default='False',
                        help="whether to take attention in context module")
    parser.add_argument('--take_alpha_loss', type=args_str2bool, default='False',
                        help="whether to take alpha factor")
    parser.add_argument('--augment_samples', type=args_str2bool, default='False',
                        help="whether to take generate sample augmentations")
    parser.add_argument('--boundary_token', type=str, default='both',
                        help="how to take features from boundary tokens, [both, left, right]")

    # boundary detection module
    parser.add_argument('--span_filter_method', type=str, default='none',
                        help="how to filter spans with their scores, [none|threshold|rate|time].")
    parser.add_argument('--filtering_strategy', type=str, default='ratio-20',
                        help="set filtering strategy with the hyper-parameters, [prop|count|ratio|threshold]-[0.1, 1, 20, 100].")
    parser.add_argument('--boundary_only_mode', type=str, default='none',
                        help="predict whether a span is an entity, [none|span|bin|bdy].")
    parser.add_argument('--boundary_det_filter_method', type=str, default='max',
                        help="the method to measure a span for filtering, [max|min]")
    parser.add_argument('--boundary_det_threshold', type=args_str2float, default='0.1',
                        help="the threshold for filtering out the span candidates")
    
    # feature fusion
    parser.add_argument('--fusion_method', type=str, default='none',
                        help="how to take the feature fusion, [none|mlp|biaffine]")

    args = fill_args(parser.parse_args())
    set_random_seeds(args.seed)

    logger.info(sys.argv)
    logger.info(args)

    # pre-define labels
    ner_label2id, ner_id2label = get_label_mapping(args)
    num_ner_labels = len(ner_label2id.keys()) + 1  # n_labels + 'O'
    
    # init model (set drop_with_punc=False for old pure)
    model = EntityModel(
        args, num_ner_labels=num_ner_labels)
    span_filter = get_span_filter(
        args, ner_label2id=ner_label2id, 
        drop_with_punc=True)  

    dev_data = Dataset(args.dev_data)
    dev_samples, dev_ner = get_samples(
        dev_data, args, span_filter=span_filter, is_training=False)
    dev_batches = batchify(dev_samples, args.eval_batch_size)

    if args.do_train:
        train_data = Dataset(args.train_data)
        train_samples, train_ner = get_samples(
            train_data, args, span_filter=span_filter, is_training=True)
        train_batches = batchify(train_samples, args.train_batch_size)

        # result logger
        best_result = 0.0

        # init optimizer and lr scheduler
        n_pos_samples = sum([
            1 if any([_l for _l in _s['spans_label'] if _l > 0]) else 0
            for _s in train_samples])
        n_batches = len(train_batches)
        if args.augment_samples:
            n_augment_samples = min(
                n_pos_samples, 
                len(train_samples)-n_pos_samples
            ) // 4
            sample_augmentor = SampleAugmentor(
                sample_file_dir=args.data_dir, 
                augment_counts=n_augment_samples)
            # approx (we don't know the real n_batches)
            n_batches += n_augment_samples * 2 // args.train_batch_size
        optimizer, scheduler = get_optimizer(
            model=model, args=args, n_batches=n_batches)
        
        tr_loss = 0
        tr_examples = 0
        global_step = 0
        best_epoch_record = None
        eval_step = len(train_batches) // args.eval_per_epoch

        for _ in tqdm(range(args.num_epoch)):
            # reload train_batches for dynamic training
            if args.augment_samples:
                # generate differently for each epoch
                augmented_train_samples = sample_augmentor(train_samples)
                train_batches = batchify(
                    augmented_train_samples + train_samples, 
                    args.train_batch_size)
                eval_step = len(train_batches) // args.eval_per_epoch
                global_step = 0
            if args.train_shuffle:
                random.shuffle(train_batches)
            for i in tqdm(range(len(train_batches))):
                output_dict = model.run_batch(
                    train_batches[i], training=True, 
                    take_alpha_labels=args.take_alpha_loss)
                loss = output_dict['ner_loss']
                # pred_prob = output_dict['ner_probs']
                loss.backward()

                tr_loss += loss.item()
                tr_examples += len(train_batches[i])
                global_step += 1

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if global_step % args.print_loss_step == 0:
                    logger.info('Epoch=%d, iter=%d, loss=%.5f'%(_, i, tr_loss / tr_examples))
                    tr_loss = 0
                    tr_examples = 0

                if global_step % eval_step == 0:
                    with torch.no_grad():
                        p, r, f1 = evaluate(model, dev_batches, dev_ner, 
                                            output_prf=True, take_alpha_labels=args.take_alpha_loss)
                    if f1 > best_result:
                        best_result = f1
                        logger.info('!!! Best valid (epoch=%d): %.2f' % (_, f1*100))
                        best_epoch_record = f"{p:.2f}/{r:.2f}/{f1:2f} on epoch {_}"
                        save_model(model, args)
        else:
            logger.info(f"BEST: {best_epoch_record}")

    if args.do_eval:
        args.bert_model_dir = args.output_dir
        model = EntityModel(args, num_ner_labels=num_ner_labels)
        if args.eval_test:  # use test data for performance evaluation (for most datasets)
            test_data = Dataset(args.test_data)
            prediction_file = os.path.join(args.output_dir, args.test_pred_filename)
        else:  # use dev data for performance evalutaion (for MSRA)
            test_data = Dataset(args.dev_data)
            prediction_file = os.path.join(args.output_dir, args.dev_pred_filename)
        test_samples, test_ner = get_samples(
            test_data, args, span_filter=span_filter, is_training=False)
        test_batches = batchify(test_samples, args.eval_batch_size)
        f1 = evaluate(model, test_batches, test_ner, 
                      take_alpha_labels=args.take_alpha_loss)
        output_ner_predictions(model, test_batches, test_data,
                               output_file=prediction_file, 
                               ner_id2label=ner_id2label)
    
    if args.inv_test:
        # inv_test_filename = 'inv_test_org.json'
        inv_test_filename = 'inv_test_all.json'
        test_data = Dataset(os.path.join(
            args.data_dir, inv_test_filename))
        inv_prediction_file = os.path.join(
            args.output_dir, f'inv_test_all_{args.test_pred_filename}')
        span_filter.include_positive = False
        bom = f"{span_filter.boundary_only_mode}"
        span_filter.boundary_only_mode = 'targeted'
        test_samples, test_ner = get_samples(
            test_data, args, span_filter=span_filter, is_training=False)
        for ts in test_samples[:5:2]:
            print(ts)
        test_batches = batchify(test_samples, args.eval_batch_size)
        f1 = evaluate(model, test_batches, test_ner, 
                      take_alpha_labels=args.take_alpha_loss)
        output_ner_predictions(model, test_batches, test_data,
                               output_file=inv_prediction_file, 
                               # output_file='/home/chendian/PURE/inv_test_on_msra_all.json', 
                               ner_id2label=ner_id2label)
        span_filter.boundary_only_mode = bom
