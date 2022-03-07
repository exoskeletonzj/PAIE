import sys
sys.path.append("../")
import copy
import spacy

from utils import _normalize_answer, find_head, hungarian_matcher, get_sentence_idx
from utils import WhitespaceTokenizer
nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)


def eval_rpf(gt_num, pred_num, correct_num):
    recall = correct_num/gt_num if gt_num!=0 else .0
    precision = correct_num/pred_num if pred_num!=0 else .0
    f1 = 2*recall*precision/(recall+precision) if (recall+precision)>1e-4 else .0
    res = {
        "recall": recall, "precision": precision, "f1": f1,
        "gt_num": gt_num, "pred_num": pred_num, "correct_num": correct_num,
    }
    return res


def eval_std_f1_score(features, invalid_gt_num=0):
    gt_num, pred_num, correct_num = 0, 0, 0
    gt_num_identify, pred_num_identify, correct_identify_num = 0, 0, 0
    
    for feature in features:
        all_pred_list = list()
        all_gt_list = list()
        for role in feature.arg_list:
            gt_list = feature.gt_dict_word[role] if role in feature.gt_dict_word else list()
            pred_list = list(set(feature.pred_dict_word[role])) if role in feature.pred_dict_word else list()
            gt_num += len(gt_list)
            pred_num += len(pred_list)
            
            for gt_span in gt_list:
                if gt_span in pred_list:
                    correct_num += 1

            all_pred_list.extend(copy.deepcopy(pred_list))
            all_gt_list.extend(gt_list)

        all_pred_list = list(set(all_pred_list))
        all_gt_list = list(set(all_gt_list))
        pred_num_identify += len(all_pred_list)
        gt_num_identify += len(all_gt_list)
        for gt_span in all_gt_list:
            if gt_span in all_pred_list:
                correct_identify_num += 1
        
    res_classification = eval_rpf(gt_num+invalid_gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num_identify+invalid_gt_num, pred_num_identify, correct_identify_num)
    return res_classification, res_identification


def eval_text_f1_score(features, invalid_gt_num=0):
    gt_num, pred_num, correct_num = 0, 0, 0
    gt_num_identify, pred_num_identify, correct_identify_num = 0, 0, 0

    for feature in features:
        all_pred_list = list()
        all_gt_list = list()
        full_text = feature.full_text
        for role in feature.arg_list:
            gt_list = feature.gt_dict_word[role] if role in feature.gt_dict_word else list()
            pred_list = list(set(feature.pred_dict_word[role])) if role in feature.pred_dict_word else list()
            #################### The only difference with eval_std_f1_score ###############################
            gt_texts = [_normalize_answer(" ".join(full_text[gt_span[0]: gt_span[1]+1])) for gt_span in gt_list]
            pred_texts = list(set([_normalize_answer(" ".join(full_text[pred_span[0]: pred_span[1]+1])) for pred_span in copy.deepcopy(pred_list)]))
            gt_list = gt_texts
            pred_list = pred_texts
            #########################################################################################################################################       
            gt_num += len(gt_list)
            pred_num += len(pred_list)
            
            for gt_span in gt_list:
                if gt_span in pred_list:
                    correct_num += 1

            all_pred_list.extend(copy.deepcopy(pred_list))
            all_gt_list.extend(gt_list)

        all_pred_list = list(set(all_pred_list))
        all_gt_list = list(set(all_gt_list))
        pred_num_identify += len(all_pred_list)
        gt_num_identify += len(all_gt_list)
        for gt_span in all_gt_list:
            if gt_span in all_pred_list:
                correct_identify_num += 1
        
    res_classification = eval_rpf(gt_num+invalid_gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num_identify+invalid_gt_num, pred_num_identify, correct_identify_num)
    return res_classification, res_identification


def eval_head_f1_score(features, invalid_gt_num=0):
    gt_num, pred_num, correct_num = 0, 0, 0
    gt_num_identify, pred_num_identify, correct_identify_num = 0, 0, 0
    last_full_text = None

    for feature in features:
        all_pred_list = list()
        all_gt_list = list()
        full_text = feature.full_text
        for role in feature.arg_list:
            gt_list = feature.gt_dict_word[role] if role in feature.gt_dict_word else list()
            pred_list = list(set(feature.pred_dict_word[role])) if role in feature.pred_dict_word else list()
            #################### The only difference with eval_std_f1_score ###############################
            full_text = feature.full_text
            if full_text!=last_full_text:
                # Reduce the time of doc generation, which is highly time-consuming
                doc = nlp(" ".join(full_text))
                last_full_text = full_text

            gt_head_texts = [str(find_head(gt_span[0], gt_span[1]+1, doc)) for gt_span in gt_list]
            pred_head_texts = list(set([str(find_head(pred_span[0], pred_span[1]+1, doc)) for pred_span in copy.deepcopy(pred_list)]))
            gt_list = gt_head_texts
            pred_list = pred_head_texts
            #########################################################################################################################################      
            gt_num += len(gt_list)
            pred_num += len(pred_list)
            
            for gt_span in gt_list:
                if gt_span in pred_list:
                    correct_num += 1

            all_pred_list.extend(copy.deepcopy(pred_list))
            all_gt_list.extend(gt_list)

        all_pred_list = list(set(all_pred_list))
        all_gt_list = list(set(all_gt_list))
        pred_num_identify += len(all_pred_list)
        gt_num_identify += len(all_gt_list)
        for gt_span in all_gt_list:
            if gt_span in all_pred_list:
                correct_identify_num += 1
        
    res_classification = eval_rpf(gt_num+invalid_gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num_identify+invalid_gt_num, pred_num_identify, correct_identify_num)
    return res_classification, res_identification


def show_results(features, output_file, metainfo):
    """ paie std show resuults """
    with open(output_file, 'w', encoding='utf-8') as f:
        for k,v in metainfo.items():
            f.write(f"{k}: {v}\n")

        for feature in features:
            example_id = feature.example_id
            
            sent = feature.enc_text
            f.write("-------------------------------------------------------------------------------------\n")
            f.write("Sent: {}\n".format(sent))
            f.write("Event type: {}\t\t\tTrigger word: {}\n".format(feature.event_type, feature.event_trigger))
            f.write("Example ID {}\n".format(example_id))
            full_text = feature.full_text
            for arg_role in feature.arg_list:
                 
                pred_list = feature.pred_dict_word[arg_role] if arg_role in feature.pred_dict_word else list()
                gt_list = feature.gt_dict_word[arg_role] if arg_role in feature.gt_dict_word else list()
                if len(pred_list)==0 and len(gt_list)==0:
                    continue
                
                if len(gt_list) == 0 and len(pred_list) > 0:
                    gt_list = [(-1,-1)] * len(pred_list)
                
                if len(gt_list) > 0 and len(pred_list) == 0:
                    pred_list = [(-1,-1)] * len(gt_list)

                gt_idxs, pred_idxs = hungarian_matcher(gt_list, pred_list)

                for pred_idx, gt_idx in zip(pred_idxs, gt_idxs):
                    if gt_list[gt_idx] == (-1,-1) and pred_list[pred_idx] == (-1,-1):
                        continue
                    else:
                        pred_text = " ".join(full_text[pred_list[pred_idx][0]: pred_list[pred_idx][1]+1]) if pred_list[pred_idx]!=(-1,-1) else "__ No answer __"
                        gt_text = " ".join(full_text[gt_list[gt_idx][0]: gt_list[gt_idx][1]+1]) if gt_list[gt_idx]!=(-1,-1) else "__ No answer __"
                    
                    if gt_list[gt_idx] == pred_list[pred_idx]:
                        f.write("Arg {} matched: Pred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role, pred_text, pred_list[pred_idx][0], pred_list[pred_idx][1], gt_text, gt_list[gt_idx][0], gt_list[gt_idx][1]))
                    else:
                        f.write("Arg {} dismatched: Pred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role, pred_text, pred_list[pred_idx][0], pred_list[pred_idx][1], gt_text, gt_list[gt_idx][0], gt_list[gt_idx][1]))
                
                if len(gt_idxs) < len(gt_list): # prediction  __no answer__
                    for idx in range(len(gt_list)):
                        if idx not in gt_idxs:
                            gt_text = " ".join(full_text[gt_list[idx][0]: gt_list[idx][1]+1])
                            f.write("Arg {} dismatched: Pred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role, "__ No answer __", -1, -1, gt_text, gt_list[idx][0], gt_list[idx][1])) 

                if len(pred_idxs) < len(pred_list): # ground truth  __no answer__
                    for idx in range(len(pred_list)):
                        if idx not in pred_idxs:
                            pred_text = " ".join(full_text[pred_list[idx][0]: pred_list[idx][1]+1])
                            f.write("Arg {} dismatched: Pred: {} ({},{})\tGt: {} ({},{})\n".format(arg_role, pred_text, pred_list[idx][0], pred_list[idx][1], "__ No answer __", -1, -1))


def eval_score_per_type(features, eval_fn, output_file):
    feature_per_type_dict = dict()
    for feature in features:
        event_type = feature.event_type
        split_feature = copy.deepcopy(feature)
        if event_type not in feature_per_type_dict:
            feature_per_type_dict[event_type] = list()
        feature_per_type_dict[event_type].append(split_feature)
    
    with open(output_file, 'w') as f:
        for event_type in sorted(feature_per_type_dict.keys()):
            res_classification, _ = eval_fn(feature_per_type_dict[event_type])
            f.write('{} : ({})\n'.format(event_type, res_classification["gt_num"]))
            f.write('EVAL: R {} P {} F {}\n'.format(res_classification["recall"], res_classification["precision"], res_classification["f1"]))
            f.write('-------------------------------------------------------------------------\n')


def eval_score_per_role(features, eval_fn, output_file):
    feature_per_role_dict = dict()
    # Enumerate all possible roles first
    for feature in features:
        for role in feature.target_info:
            if role not in feature_per_role_dict:
                feature_per_role_dict[role] = list()
            split_feature = copy.deepcopy(feature)
            new_pred_dict = {r:split_feature.pred_dict_word[r] if r==role else list() for r in split_feature.pred_dict_word}
            split_feature.pred_dict_word = new_pred_dict
            new_gt_dict = {r:split_feature.gt_dict_word[r] if r==role else list() for r in split_feature.gt_dict_word}
            split_feature.gt_dict_word = new_gt_dict
            feature_per_role_dict[role].append(split_feature)

    with open(output_file, 'w') as f:
        for role_type in sorted(feature_per_role_dict.keys()):
            res_classification, _ = eval_fn(feature_per_role_dict[role_type])
            f.write('{} : ({})\n'.format(role_type, res_classification["gt_num"]))
            f.write('EVAL: R {} P {} F {}\n'.format(res_classification["recall"], res_classification["precision"], res_classification["f1"]))
            f.write('-------------------------------------------------------------------------\n')


def eval_score_per_argnum(features, eval_fn, output_file):

    def get_split_feature(feature, valid_argnum):
        def f(x):
            if x<=1:
                return 1
            elif x>=4:
                return 4
            else:
                return x
                
        split_feature = copy.deepcopy(feature)
        for role, span_list in split_feature.gt_dict_word.items():
            arg_num = len([span for span in span_list if span!=(-1, -1)])
            # for span in split_feature.gt_dict_word[role]:
            #     if span==(-1, -1): 
            #         continue
            #     arg_num += 1
            if f(arg_num) != valid_argnum:
                split_feature.gt_dict_word[role] = []
                split_feature.pred_dict_word[role] = []
        return split_feature

    with open(output_file, 'w') as f:
        for argnum in [1, 2, 3, 4]:
            split_features = []
            for feature in features:
                split_features.append(get_split_feature(feature, argnum))
            res_classification, _ = eval_fn(split_features)
            f.write("ARGNUM:{} ({})\n".format(argnum, res_classification["gt_num"]))
            f.write('EVAL: R {} P {} F {}\n'.format(res_classification["recall"], res_classification["precision"], res_classification["f1"]))
            f.write('-------------------------------------------------------------------------\n')


def eval_score_per_dist(features, examples, eval_fn, output_file):

    def get_split_feature(feature, first_word_locs, valid_range):
        if isinstance(valid_range, int):
            valid_range = [valid_range]
        split_feature = copy.deepcopy(feature)
        trigger_loc = get_sentence_idx(first_word_locs, feature.event_trigger[1][0])

        for role in split_feature.target_info:
            delete_idx_list = list()
            if role in split_feature.pred_dict_word:
                for idx, span in enumerate(split_feature.pred_dict_word[role]):
                    dist = get_sentence_idx(first_word_locs, span[0]) - trigger_loc
                    if dist not in valid_range:
                        delete_idx_list.append(idx)
                split_feature.pred_dict_word[role] = [v for idx, v in enumerate(split_feature.pred_dict_word[role]) if idx not in delete_idx_list]

            delete_idx_list = list()
            if role in split_feature.gt_dict_word:
                for idx, span in enumerate(split_feature.gt_dict_word[role]):
                    dist = get_sentence_idx(first_word_locs, span[0]) - trigger_loc
                    if dist not in valid_range:
                        delete_idx_list.append(idx)
                split_feature.gt_dict_word[role] = [v for idx, v in enumerate(split_feature.gt_dict_word[role]) if idx not in delete_idx_list]

        return split_feature

    example_dict = {example.doc_id:example for example in examples}
    with open(output_file, 'w') as f:
        for dist in [-2, -1, 0, 1, 2]:
            split_features = []
            for feature in features:
                example = example_dict[feature.example_id]
                first_word_locs = example.first_word_locs
                split_features.append(get_split_feature(feature, first_word_locs, dist))
            res_classification, _ = eval_fn(split_features)
            f.write("Dist:{} ({})\n".format(dist, res_classification["gt_num"]))
            f.write('EVAL: R {} P {} F {}'.format(res_classification["recall"], res_classification["precision"], res_classification["f1"]))
            f.write('-------------------------------------------------------------------------\n')