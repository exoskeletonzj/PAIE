from gc import collect
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"
if os.environ.get('DEBUG', False): print('\033[92m'+'Running code in DEBUG mode'+'\033[0m')
import os.path as osp
import logging

import ipdb
import copy
import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter

import sys
from models import build_model
from processors import build_processor

from utils import set_seed, get_best_indexes, get_best_index, count_time,  show_results
from utils import eval_score_std_span_full_metrics, eval_score_per_type, eval_score_per_role, eval_score_per_dist, eval_score_per_argnum
from runner.runner import Runner

logger = logging.getLogger(__name__)

def train(args, model, processor):
    set_seed(args)

    logger.info("train dataloader generation")
    _, train_features, train_dataloader = processor.generate_dataloader('train')
    logger.info("dev dataloader generation")
    dev_examples, dev_features, dev_dataloader = processor.generate_dataloader('dev')
    logger.info("test dataloader generation")
    test_examples, test_features, test_dataloader = processor.generate_dataloader('test')

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*args.warmup_steps, num_training_steps=args.max_steps)

    # Train!
    tb_writer = SummaryWriter(os.path.join(args.output_dir, 'event'))
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader)*args.batch_size)
    logger.info("  batch size = %d", args.batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)


    runner = Runner(
        cfg=args,
        data_samples=[None, None, None],
        data_features=[train_features, dev_features, test_features],
        data_loaders=[train_dataloader, dev_dataloader, test_dataloader],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_fn_dict=None,
    )
    runner.run()

#     global_step = 0
#     tr_loss, smooth_loss = 0.0, 0.0
#     best_f1_dev, best_f1_test, related_f1_test = 0.0, 0.0, 0.0

#     model.zero_grad()
#     while global_step <= args.max_steps:
#         for step, batch in enumerate(train_dataloader):
#             model.train()
#             inputs = {
#                 'enc_input_ids':  batch[0].to(args.device), 
#                 'enc_mask_ids':   batch[1].to(args.device), 
#                 'arg_list':       batch[9],
#             }
#             if args.model_type == 'base':
#                 inputs.update({
#                 'decoder_prompt_ids_list':      [item.to(args.device) for item in batch[2]], 
#                 'decoder_prompt_mask_list': [item.to(args.device) for item in batch[3]],
#                 'decoder_prompt_start_positions_list': [item.to(args.device) for item in batch[12]],
#                 'decoder_prompt_end_positions_list': [item.to(args.device) for item in batch[13]],
#                 'start_position_ids': [item.to(args.device) for item in batch[14]],
#                 'end_position_ids': [item.to(args.device) for item in batch[15]],
#                 })
#             if "paie" in args.model_type:
#                 inputs.update({
#                 'dec_prompt_ids':           batch[4].to(args.device),
#                 'dec_prompt_mask_ids':      batch[5].to(args.device),
#                 'target_info':              batch[6], 
#                 'old_tok_to_new_tok_indexs':batch[7],
#                 'arg_joint_prompts':        batch[8],
#                 })

#             loss, _= model(**inputs)
#             if args.gradient_accumulation_steps > 1:
#                 loss = loss / args.gradient_accumulation_steps

#             loss.backward()

#             if args.max_grad_norm != 0:
#                 nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

#             tr_loss += loss.item()
#             smooth_loss += loss.item()/args.logging_steps
#             if (step+1)%args.gradient_accumulation_steps==0:
#                 optimizer.step()
#                 scheduler.step()
#                 model.zero_grad()
#                 global_step += 1

#             if global_step % args.logging_steps == 0:
#                 logging.info("-----------------------global_step: {} -------------------------------- ".format(global_step))
#                 logging.info('lr: {}'.format(scheduler.get_lr()[0]))
#                 tb_writer.add_scalar('smooth_loss', smooth_loss, global_step)
#                 logging.info('smooth_loss: {}'.format(smooth_loss))
#                 smooth_loss = .0

#             if global_step % args.eval_steps == 0:
#                 [dev_r, dev_p, dev_f1,_,_,_], [dev_r_text, dev_p_text, dev_f1_text, _,_,_], dev_original_features = \
#                     evaluate(args, model, dev_features, dev_dataloader, processor.tokenizer, set_type='dev')
#                 [test_r, test_p, test_f1,_,_,_], [test_r_text, test_p_text, test_f1_text,_,_,_], test_original_features = \
#                     evaluate(args, model, test_features, test_dataloader,  processor.tokenizer, set_type='test')

#                 tb_writer.add_scalar('dev_f1', dev_f1, global_step)
#                 tb_writer.add_scalar('dev_f1_text', dev_f1_text, global_step)
#                 tb_writer.add_scalar('test_f1', test_f1, global_step)
#                 tb_writer.add_scalar('test_f1_text', test_f1_text, global_step)

#                 output_dir = os.path.join(args.output_dir, 'checkpoint')
#                 os.makedirs(output_dir, exist_ok=True)

#                 if test_f1 > best_f1_test:
#                     best_f1_test = test_f1
#                     show_results(test_features, os.path.join(args.output_dir, f'best_test_results.log'),
#                         {"test related best score": f"P: {test_p} R: {test_r} f1: {test_f1}", "global step": global_step}
#                     )

#                 if dev_f1 > best_f1_dev:
#                     best_f1_dev = dev_f1
#                     related_f1_test = test_f1
#                     show_results(test_features, os.path.join(args.output_dir, f'best_test_related_results.log'), 
#                         {"test related best score": f"P: {test_p} R: {test_r} f1: {test_f1}", "global step": global_step}
#                     )
#                     show_results(dev_features, os.path.join(args.output_dir, f'best_dev_results.log'), 
#                         {"dev best score": f"P: {dev_p} R: {dev_r} f1: {dev_f1}", "global step": global_step}
#                     )
#                     eval_score_per_type(test_original_features, args.dataset_type, 
#                         os.path.join(args.output_dir, f'results_per_type.txt'), 
#                     )
#                     eval_score_per_role(test_original_features, args.dataset_type, 
#                         os.path.join(args.output_dir, f'results_per_role.txt'), 
#                     )
#                     if args.dataset_type=='ace_eeqa':
#                         eval_score_per_argnum(dev_original_features, args.dataset_type, 
#                             os.path.join(args.output_dir, f'dev_results_per_argnum.txt'), 
#                         )
#                         eval_score_per_argnum(test_original_features, args.dataset_type, 
#                             os.path.join(args.output_dir, f'test_results_per_argnum.txt'), 
#                         )
#                     else:
#                         eval_score_per_dist(dev_original_features, dev_examples, args.dataset_type, 
#                             os.path.join(args.output_dir, f'dev_results_per_dist.txt'), 
#                         )
#                         eval_score_per_dist(test_original_features, test_examples, args.dataset_type, 
#                             os.path.join(args.output_dir, f'test_results_per_dist.txt'), 
#                         )
#                     model.save_pretrained(output_dir)

#                 tb_writer.add_scalar('best_f1_dev', best_f1_dev, global_step)
#                 tb_writer.add_scalar('best_f1_test', best_f1_test, global_step)
#                 tb_writer.add_scalar('related_f1_test', related_f1_test, global_step)

#                 logging.info('current best dev-f1 score: {}'.format(best_f1_dev))
#                 logging.info('current related test-f1 score: {}'.format(related_f1_test))
#                 logging.info('current best test-f1 score: {}'.format(best_f1_test))
  
#     tr_loss /= global_step
#     # tb_writer.close()


# @count_time
# def calculate(args, model, features, dataloader):
#     feature_id_list, role_list = [], []
#     full_start_logit_list, full_end_logit_list = [], []

#     for batch in dataloader:
#         model.eval()
#         with torch.no_grad(): 
#             inputs = {
#                 'enc_input_ids':  batch[0].to(args.device), 
#                 'enc_mask_ids':   batch[1].to(args.device), 
#                 'arg_list':       batch[9],
#             }
#             if args.model_type == 'base':
#                 inputs.update({
#                     'decoder_prompt_ids_list':      [item.to(args.device) for item in batch[2]], 
#                     'decoder_prompt_mask_list': [item.to(args.device) for item in batch[3]],
#                     'decoder_prompt_start_positions_list': [item.to(args.device) for item in batch[12]],
#                     'decoder_prompt_end_positions_list': [item.to(args.device) for item in batch[13]],
#                 })
#             if "paie" in args.model_type:
#                 inputs.update({
#                     'dec_prompt_ids':           batch[4].to(args.device),
#                     'dec_prompt_mask_ids':      batch[5].to(args.device),
#                     'old_tok_to_new_tok_indexs':batch[7],
#                     'arg_joint_prompts':        batch[8],
#                     'target_info':              None, 
#                 })

#             _, outputs_list = model(**inputs)

#         bs = len(batch[0])
#         for i in range(bs):
#             predictions = outputs_list[i]
#             feature_id = batch[11][i].item()
#             feature = features[feature_id]
#             feature.init_pred()
#             feature.set_gt(args.model_type)
#             for arg_role in batch[9][i]:
#                 [start_logits_list, end_logits_list] = predictions[arg_role] # NOTE base model should also has these kind of output
#                 feature.pred_dict[arg_role] = list()
#                 for (start_logit, end_logit) in zip(start_logits_list, end_logits_list):
#                     feature_id_list.append(feature_id)
#                     role_list.append(arg_role)
#                     full_start_logit_list.append(start_logit)
#                     full_end_logit_list.append(end_logit)
    
#     return feature_id_list, role_list, full_start_logit_list, full_end_logit_list


# def evaluate(args, model, features, dataloader, tokenizer, set_type='dev'):
#     kwargs = {"args": args, "model": model, "features":features, "dataloader":dataloader}
#     feature_id_list, role_list, full_start_logit_list, full_end_logit_list = calculate(**kwargs)

#     pred_list = []
#     if "paie" in args.model_type:
#         for s in range(0, len(full_start_logit_list), args.infer_batch_size):
#             sub_max_locs, cal_time, mask_time, score_time = get_best_indexes(features, feature_id_list[s:s+args.infer_batch_size], \
#                 full_start_logit_list[s:s+args.infer_batch_size], full_end_logit_list[s:s+args.infer_batch_size], args)
#             pred_list.extend(sub_max_locs)
#         for (pred, feature_id, role) in zip(pred_list, feature_id_list, role_list):
#             features[feature_id].pred_dict[role].append(\
#                 (pred[0].item(), pred[1].item())
#             )
#     else:
#         for feature_id, role, start_logit, end_logit in zip(feature_id_list, role_list, full_start_logit_list, full_end_logit_list):
#             feature = features[feature_id]
#             answer_span_pred_list = get_best_index(feature, start_logit, end_logit, args.max_span_length, args.max_span_num, args.th_delta)
#             feature.pred_dict[role] = answer_span_pred_list
    
#     original_features = copy.deepcopy(features)     # After eval_score, the span recorded in features will be changed. We want to keep the original value for further evaluation.
#     perf_span, perf_text, perf_identify, perf_head = eval_score_std_span_full_metrics(features, args.dataset_type)
#     logging.info('SPAN-EVAL {} ({}): R {} P {} F {}'.format(set_type, perf_span[3], perf_span[0], perf_span[1], perf_span[2]))
#     logging.info('TEXT-EVAL {} ({}): R {} P {} F {}'.format(set_type, perf_text[3], perf_text[0], perf_text[1], perf_text[2]))
#     logging.info('IDEN-EVAL {} ({}): R {} P {} F {}'.format(set_type, perf_identify[3], perf_identify[0], perf_identify[1], perf_identify[2]))
#     logging.info('HEAD-EVAL {} ({}): R {} P {} F {}'.format(set_type, perf_head[3], perf_head[0], perf_head[1], perf_head[2]))

#     return perf_span, perf_text, original_features


def main():
    from config_parser import get_args_parser
    args = get_args_parser()

    
    print(f"Output full path {osp.join(os.getcwd(), args.output_dir)}")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(
        filename=os.path.join(args.output_dir, "log.txt"), \
        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO
        )
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    set_seed(args)

    model, tokenizer, config = build_model(args, args.model_type) 
    model.to(args.device)

    processor = build_processor(args, tokenizer)

    logger.info("Training/evaluation parameters %s", args)
    train(args, model, processor)
            

if __name__ == "__main__":
    main()