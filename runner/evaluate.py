import copy
import torch
import logging
from utils import get_best_indexes, get_best_index
from utils import eval_score_std_span_full_metrics

logger = logging.getLogger(__name__)


class BaseEvaluator:
    def __init__(
        self,
        cfg=None,
        data_loader=None,
        model=None,
        metric_fn_dict=None,
    ):

        self.cfg = cfg
        self.eval_loader = data_loader
        self.model = model
        self._init_metric(metric_fn_dict)

    
    def _init_metric(self, metric_fn_dict):
        self.metric_fn_dict = metric_fn_dict
        # self.metric_val_dict = {metric:None for metric in metric_fn_dict}


    def calculate_one_batch(self, batch):
        inputs, named_v = self.convert_batch_to_inputs(batch)
        with torch.no_grad():
            _, outputs_list = self.model(**inputs)
        return outputs_list, named_v


    def evaluate_one_batch(self, batch):
        outputs_list, named_v = self.calculate_one_batch(batch)
        self.collect_fn(outputs_list, named_v, batch)


    def evaluate(self):
        self.model.eval()
        self.build_and_clean_record()
        for batch in self.eval_loader:
            self.evaluate_one_batch(batch)
        self.predict()


    def build_and_clean_record(self):
        raise NotImplementedError


    def collect_fn(self, outputs_list, named_v, batch):
        raise NotImplementedError

     
    def convert_batch_to_inputs(self, batch):
        return NotImplementedError


    def predict(self):
        raise NotImplementedError


class Evaluator(BaseEvaluator):
    def __init__(
        self, 
        cfg=None, 
        data_loader=None, 
        model=None, 
        metric_fn_dict=None,
        features=None,
    ):
        super().__init__(cfg, data_loader, model, metric_fn_dict)
        self.features = features

    
    def convert_batch_to_inputs(self, batch):
        inputs = {
            'enc_input_ids':  batch[0].to(self.cfg.device), 
            'enc_mask_ids':   batch[1].to(self.cfg.device), 
            'dec_prompt_ids':           batch[4].to(self.cfg.device),
            'dec_prompt_mask_ids':      batch[5].to(self.cfg.device),
            'old_tok_to_new_tok_indexs':batch[7],
            'arg_joint_prompts':        batch[8],
            'target_info':              None, 
            'arg_list':       batch[9],
        }
        named_v = {
            "arg_roles": batch[9],
            "feature_ids": batch[11],
        }
        return inputs, named_v


    def build_and_clean_record(self):
        self.record = {
            "feature_id_list": list(),
            "role_list": list(),
            "full_start_logit_list": list(),
            "full_end_logit_list": list()
        }


    def collect_fn(self, outputs_list, named_v, batch):   
        bs = len(batch[0])
        for i in range(bs):
            predictions = outputs_list[i]
            feature_id = named_v["feature_ids"][i].item()
            feature = self.features[feature_id]
            feature.init_pred()
            feature.set_gt(self.cfg.model_type)
            for arg_role in named_v["arg_roles"][i]:
                [start_logits_list, end_logits_list] = predictions[arg_role] # NOTE base model should also has these kind of output
                feature.pred_dict[arg_role] = list()
                for (start_logit, end_logit) in zip(start_logits_list, end_logits_list):
                    self.record["feature_id_list"].append(feature_id)
                    self.record["role_list"].append(arg_role)
                    self.record["full_start_logit_list"].append(start_logit)
                    self.record["full_end_logit_list"].append(end_logit)

    
    def predict(self):
        pred_list = []
        if "paie" in self.cfg.model_type:
            for s in range(0, len(self.record["full_start_logit_list"]), self.cfg.infer_batch_size):
                sub_max_locs, cal_time, mask_time, score_time = get_best_indexes(self.features, self.record["feature_id_list"][s:s+self.cfg.infer_batch_size], \
                    self.record["full_start_logit_list"][s:s+self.cfg.infer_batch_size], self.record["full_end_logit_list"][s:s+self.cfg.infer_batch_size], self.cfg)
                pred_list.extend(sub_max_locs)
            for (pred, feature_id, role) in zip(pred_list, self.record["feature_id_list"], self.record["role_list"]):
                self.features[feature_id].pred_dict[role].append((pred[0].item(), pred[1].item()))
        else:
            for feature_id, role, start_logit, end_logit in zip(
                self.record["feature_id_list"], self.record["role_list"], self.record["full_start_logit_list"], self.record["full_end_logit_list"]
            ):
                feature = self.features[feature_id]
                answer_span_pred_list = get_best_index(feature, start_logit, end_logit, self.cfg.max_span_length, self.cfg.max_span_num, self.cfg.th_delta)
                feature.pred_dict[role] = answer_span_pred_list

        set_type = "None"
        original_features = copy.deepcopy(self.features)     # After eval_score, the span recorded in features will be changed. We want to keep the original value for further evaluation.
        perf_span, perf_text, perf_identify, perf_head = eval_score_std_span_full_metrics(self.features, self.cfg.dataset_type)
        logger.info('SPAN-EVAL {} ({}): R {} P {} F {}'.format(set_type, perf_span[3], perf_span[0], perf_span[1], perf_span[2]))
        logger.info('TEXT-EVAL {} ({}): R {} P {} F {}'.format(set_type, perf_text[3], perf_text[0], perf_text[1], perf_text[2]))
        logger.info('IDEN-EVAL {} ({}): R {} P {} F {}'.format(set_type, perf_identify[3], perf_identify[0], perf_identify[1], perf_identify[2]))
        logger.info('HEAD-EVAL {} ({}): R {} P {} F {}'.format(set_type, perf_head[3], perf_head[0], perf_head[1], perf_head[2]))

        # self.metric_val_dict["perf_span"] = perf_span
        # self.metric_val_dict["perf_text"] = perf_text
        # return perf_span, perf_text, original_features
        

        
