import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"
if os.environ.get('DEBUG', False): print('\033[92m'+'Running code in DEBUG mode'+'\033[0m')
import os.path as osp

import ipdb
from tqdm import tqdm
from models import build_model
from processors import build_processor
from engine import calculate
from copy import deepcopy
from utils import set_seed, get_best_indexes, get_best_index, eval_score_std_span, get_sentence_idx, _new_tok_id2old_tok_id, _tok_idx2word_idx

import logging
logger = logging.getLogger(__name__)


def get_split_feature(feature, valid_argnum):
    def f(x):
        if x<=1:
            return 1
        elif x>=4:
            return 4
        else:
            return x

    split_feature = deepcopy(feature)
    for role in split_feature.target_info:
        arg_num = 0
        for idx, span in enumerate(split_feature.gt_dict[role]):
            if span==(0, 0): 
                continue
            arg_num += 1
        if f(arg_num) != valid_argnum:
            split_feature.gt_dict[role] = []
            split_feature.pred_dict[role] = []

    return split_feature

        
def evaluate(args, model, examples, features, dataloader, tokenizer, set_type='dev'):
    feature_id_list, role_list, full_start_logit_list, full_end_logit_list = calculate(args, model, features, dataloader)

    pred_list = []
    if "paie" in args.model_type:
        for s in range(0, len(full_start_logit_list), args.infer_batch_size):
            sub_max_locs, cal_time, mask_time, score_time = get_best_indexes(features, feature_id_list[s:s+args.infer_batch_size], \
                full_start_logit_list[s:s+args.infer_batch_size], full_end_logit_list[s:s+args.infer_batch_size], args)
            pred_list.extend(sub_max_locs)
        for (pred, feature_id, role) in zip(pred_list, feature_id_list, role_list):
            features[feature_id].pred_dict[role].append(\
                (pred[0].item(), pred[1].item())
            )
    else:
        for feature_id, role, start_logit, end_logit in zip(feature_id_list, role_list, full_start_logit_list, full_end_logit_list):
            feature = features[feature_id]
            answer_span_pred_list = get_best_index(feature, start_logit, end_logit, args.max_span_length, args.max_span_num, args.th_delta)
            feature.pred_dict[role] = answer_span_pred_list
    
    example_dict = {example.doc_id:example for example in examples}
    for argnum in [1, 2, 3, 4]:
        logger.info("argnum:{}".format(argnum))
        split_features = []
        for feature in tqdm(features):
            split_features.append(get_split_feature(feature, argnum))

        perf_span, perf_text = eval_score_std_span(split_features, args.dataset_type)
        logging.info('SPAN-EVAL {} ({}): R {} P {} F {}'.format(set_type, perf_span[3], perf_span[0], perf_span[1], perf_span[2]))
        logging.info('TEXT-EVAL {} ({}): R {} P {} F {}'.format(set_type, perf_text[3], perf_text[0], perf_text[1], perf_text[2]))

    return perf_span, perf_text


def main():
    from config_parser import get_args_parser
    args = get_args_parser()

    logging.basicConfig(
        format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
        datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO
        )
    set_seed(args)

    model, tokenizer, config = build_model(args, args.model_type) 
    model.to(args.device)

    processor = build_processor(args, tokenizer)

    logger.info("test dataloader generation")
    test_examples, test_features, test_dataloader = processor.generate_dataloader('dev')

    evaluate(args, model, test_examples, test_features, test_dataloader, tokenizer, set_type='dev')
    
            

if __name__ == "__main__":
    main()


# 100%|██████████| 450/450 [00:00<00:00, 546.34it/s]
# 11/07/2021 11:30:18 - INFO - root -   SPAN-EVAL dev (506): R 0.717391304347826 P 0.7393075356415478 F 0.7281845536609829
# 11/07/2021 11:30:18 - INFO - root -   TEXT-EVAL dev (506): R 0.7193675889328063 P 0.7413441955193483 F 0.7301905717151453
# 11/07/2021 11:30:18 - INFO - __main__ -   argnum:2
# 100%|██████████| 450/450 [00:00<00:00, 800.80it/s]
# 11/07/2021 11:30:19 - INFO - root -   SPAN-EVAL dev (66): R 0.5606060606060606 P 0.9487179487179487 F 0.7047619047619047
# 11/07/2021 11:30:19 - INFO - root -   TEXT-EVAL dev (66): R 0.5606060606060606 P 0.9487179487179487 F 0.7047619047619047
# 11/07/2021 11:30:19 - INFO - __main__ -   argnum:3
# 100%|██████████| 450/450 [00:00<00:00, 817.34it/s]
# 11/07/2021 11:30:20 - INFO - root -   SPAN-EVAL dev (21): R 0.5714285714285714 P 1.0 F 0.7272727272727273
# 11/07/2021 11:30:20 - INFO - root -   TEXT-EVAL dev (21): R 0.5714285714285714 P 1.0 F 0.7272727272727273
# 11/07/2021 11:30:20 - INFO - __main__ -   argnum:4
# 100%|██████████| 450/450 [00:00<00:00, 581.33it/s]
# 11/07/2021 11:30:20 - INFO - root -   SPAN-EVAL dev (12): R 0.08333333333333333 P 1.0 F 0.15384615384615385
# 11/07/2021 11:30:20 - INFO - root -   TEXT-EVAL dev (12): R 0.08333333333333333 P 1.0 F 0.15384615384615385



# 100%|██████████| 450/450 [00:02<00:00, 205.99it/s]
# 11/06/2021 21:20:03 - INFO - root -   SPAN-EVAL dev (506): R 0.7114624505928854 P 0.7422680412371134 F 0.7265388496468214
# 11/06/2021 21:20:03 - INFO - root -   TEXT-EVAL dev (506): R 0.7134387351778656 P 0.7443298969072165 F 0.7285570131180624
# 11/06/2021 21:20:03 - INFO - __main__ -   argnum:2
# 100%|██████████| 450/450 [00:02<00:00, 189.27it/s]
# 11/06/2021 21:20:06 - INFO - root -   SPAN-EVAL dev (66): R 0.36363636363636365 P 0.8888888888888888 F 0.5161290322580644
# 11/06/2021 21:20:06 - INFO - root -   TEXT-EVAL dev (66): R 0.36363636363636365 P 0.8888888888888888 F 0.5161290322580644
# 11/06/2021 21:20:06 - INFO - __main__ -   argnum:3
# 100%|██████████| 450/450 [00:02<00:00, 206.21it/s]
# 11/06/2021 21:20:08 - INFO - root -   SPAN-EVAL dev (21): R 0.2857142857142857 P 1.0 F 0.4444444444444445
# 11/06/2021 21:20:08 - INFO - root -   TEXT-EVAL dev (21): R 0.2857142857142857 P 1.0 F 0.4444444444444445
# 11/06/2021 21:20:08 - INFO - __main__ -   argnum:4
# 100%|██████████| 450/450 [00:02<00:00, 183.52it/s]
# 11/06/2021 21:20:10 - INFO - root -   SPAN-EVAL dev (12): R 0.08333333333333333 P 0.5 F 0.14285714285714285
# 11/06/2021 21:20:10 - INFO - root -   TEXT-EVAL dev (12): R 0.08333333333333333 P 0.5 F 0.14285714285714285