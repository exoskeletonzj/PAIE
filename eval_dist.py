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


def get_split_feature(feature, first_word_locs, valid_range=[-2, -1, 0, 1, 2]):
    if isinstance(valid_range, int):
        valid_range = [valid_range]
    split_feature = deepcopy(feature)

    new_tok_index_to_old_tok_index = _new_tok_id2old_tok_id(feature.old_tok_to_new_tok_index)
    offset = feature.event_trigger[2]

    trigger_loc = get_sentence_idx(first_word_locs, feature.event_trigger[1][0])
    for role in split_feature.target_info:
        delete_idx_list = list()
        for idx, span in enumerate(split_feature.pred_dict[role]):
            if span==(0, 0): 
                continue
            new_span = _tok_idx2word_idx(span, new_tok_index_to_old_tok_index, offset)
            dist = get_sentence_idx(first_word_locs, new_span[0]) - trigger_loc
            if dist not in valid_range:
                delete_idx_list.append(idx)
        split_feature.pred_dict[role] = [v for idx, v in enumerate(split_feature.pred_dict[role]) if idx not in delete_idx_list]

        delete_idx_list = list()
        for idx, span in enumerate(split_feature.gt_dict[role]):
            if span==(0, 0): 
                continue
            new_span = _tok_idx2word_idx(span, new_tok_index_to_old_tok_index, offset)
            dist = get_sentence_idx(first_word_locs, new_span[0]) - trigger_loc
            if dist not in valid_range:
                delete_idx_list.append(idx)
        split_feature.gt_dict[role] = [v for idx, v in enumerate(split_feature.gt_dict[role]) if idx not in delete_idx_list]

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
    for dist in [-2, -1, 0, 1, 2]:
        logger.info("dist:{}".format(dist))
        split_features = []
        for feature in tqdm(features):
            example = example_dict[feature.example_id]
            first_word_locs = example.first_word_locs
            split_features.append(get_split_feature(feature, first_word_locs, dist))

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


# 100%|██████████| 924/924 [00:01<00:00, 487.06it/s]
# 11/06/2021 18:35:26 - INFO - root -   SPAN-EVAL dev (79): R 0.22784810126582278 P 0.28125 F 0.2517482517482517
# 11/06/2021 18:35:26 - INFO - root -   TEXT-EVAL dev (79): R 0.25316455696202533 P 0.3125 F 0.27972027972027974
# 11/06/2021 18:35:26 - INFO - __main__ -   dist:-1
# 100%|██████████| 924/924 [00:02<00:00, 436.02it/s]
# 11/06/2021 18:35:28 - INFO - root -   SPAN-EVAL dev (164): R 0.27439024390243905 P 0.3191489361702128 F 0.29508196721311475
# 11/06/2021 18:35:28 - INFO - root -   TEXT-EVAL dev (164): R 0.29878048780487804 P 0.3475177304964539 F 0.32131147540983607
# 11/06/2021 18:35:28 - INFO - __main__ -   dist:0
# 100%|██████████| 924/924 [00:01<00:00, 498.47it/s]
# 11/06/2021 18:35:30 - INFO - root -   SPAN-EVAL dev (1811): R 0.5289895085588073 P 0.5334075723830735 F 0.5311893540338231
# 11/06/2021 18:35:30 - INFO - root -   TEXT-EVAL dev (1811): R 0.5599116510215351 P 0.5645879732739421 F 0.5622400887163846
# 11/06/2021 18:35:30 - INFO - __main__ -   dist:1
# 100%|██████████| 924/924 [00:01<00:00, 495.06it/s]
# 11/06/2021 18:35:32 - INFO - root -   SPAN-EVAL dev (87): R 0.2413793103448276 P 0.27631578947368424 F 0.2576687116564418
# 11/06/2021 18:35:32 - INFO - root -   TEXT-EVAL dev (87): R 0.2413793103448276 P 0.27631578947368424 F 0.2576687116564418
# 11/06/2021 18:35:32 - INFO - __main__ -   dist:2
# 100%|██████████| 924/924 [00:01<00:00, 579.16it/s]
# 11/06/2021 18:35:34 - INFO - root -   SPAN-EVAL dev (47): R 0.19148936170212766 P 0.391304347826087 F 0.2571428571428572
# 11/06/2021 18:35:34 - INFO - root -   TEXT-EVAL dev (47): R 0.19148936170212766 P 0.391304347826087 F 0.2571428571428572


# 100%|██████████| 924/924 [00:04<00:00, 195.47it/s]
# 11/06/2021 18:45:51 - INFO - root -   SPAN-EVAL dev (79): R 0.11392405063291139 P 0.2647058823529412 F 0.1592920353982301
# 11/06/2021 18:45:51 - INFO - root -   TEXT-EVAL dev (79): R 0.12658227848101267 P 0.29411764705882354 F 0.1769911504424779
# 11/06/2021 18:45:51 - INFO - __main__ -   dist:-1
# 100%|██████████| 924/924 [00:04<00:00, 193.83it/s]
# 11/06/2021 18:45:56 - INFO - root -   SPAN-EVAL dev (164): R 0.18902439024390244 P 0.3333333333333333 F 0.24124513618677043
# 11/06/2021 18:45:56 - INFO - root -   TEXT-EVAL dev (164): R 0.21341463414634146 P 0.3763440860215054 F 0.27237354085603116
# 11/06/2021 18:45:56 - INFO - __main__ -   dist:0
# 100%|██████████| 924/924 [00:04<00:00, 200.28it/s]
# 11/06/2021 18:46:01 - INFO - root -   SPAN-EVAL dev (1805): R 0.48476454293628807 P 0.5573248407643312 F 0.5185185185185185
# 11/06/2021 18:46:01 - INFO - root -   TEXT-EVAL dev (1805): R 0.5102493074792244 P 0.586624203821656 F 0.5457777777777778
# 11/06/2021 18:46:01 - INFO - __main__ -   dist:1
# 100%|██████████| 924/924 [00:04<00:00, 198.63it/s]
# 11/06/2021 18:46:06 - INFO - root -   SPAN-EVAL dev (87): R 0.1839080459770115 P 0.34782608695652173 F 0.24060150375939848
# 11/06/2021 18:46:06 - INFO - root -   TEXT-EVAL dev (87): R 0.1839080459770115 P 0.34782608695652173 F 0.24060150375939848
# 11/06/2021 18:46:06 - INFO - __main__ -   dist:2
# 100%|██████████| 924/924 [00:04<00:00, 216.54it/s]
# 11/06/2021 18:46:10 - INFO - root -   SPAN-EVAL dev (47): R 0.0851063829787234 P 0.36363636363636365 F 0.13793103448275862
# 11/06/2021 18:46:10 - INFO - root -   TEXT-EVAL dev (47): R 0.0851063829787234 P 0.36363636363636365 F 0.13793103448275862