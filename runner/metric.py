import sys
sys.path.append("../")
import copy
import spacy

from utils import _normalize_answer, find_head
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


def eval_std_f1_score(features):
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
        
    res_classification = eval_rpf(gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num_identify, pred_num_identify, correct_identify_num)
    return res_classification, res_identification


def eval_text_f1_score(features):
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
        
    res_classification = eval_rpf(gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num_identify, pred_num_identify, correct_identify_num)
    return res_classification, res_identification


def eval_head_f1_score(features):
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
        
    res_classification = eval_rpf(gt_num, pred_num, correct_num)
    res_identification = eval_rpf(gt_num_identify, pred_num_identify, correct_identify_num)
    return res_classification, res_identification