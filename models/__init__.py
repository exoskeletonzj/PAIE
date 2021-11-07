import sys
sys.path.append("../")
import logging
from transformers import BartConfig, BartTokenizerFast
from tokenizers import AddedToken
from .paie import PAIE
from .single_prompt import BartSingleArg
from utils import read_prompt_group

MODEL_CLASSES = {
    'paie': (BartConfig, PAIE, BartTokenizerFast),
    'base': (BartConfig, BartSingleArg, BartTokenizerFast)
}

logger = logging.getLogger(__name__)

def build_model(args, model_type):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    config = config_class.from_pretrained(args.model_name_or_path)
    config.model_name_or_path = args.model_name_or_path
    config.device = args.device
    config.prompt_context = args.prompt_context

    # length
    config.max_enc_seq_length = args.max_enc_seq_length
    config.max_dec_seq_length= args.max_dec_seq_length
    config.max_prompt_seq_length=args.max_prompt_seq_length
    config.max_span_length = args.max_span_length

    config.matching_method_train = args.matching_method_train

    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, add_special_tokens=True)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    # Add trigger special tokens and continuous token (maybe in prompt)
    new_token_list = ['<t>', '</t>']
    prompts = read_prompt_group(args.prompt_path)
    for event_type, prompt in prompts.items():
        token_list = prompt.split()
        for token in token_list:
            if token.startswith('<') and token.endswith('>') and token not in new_token_list:
                new_token_list.append(token)
    tokenizer.add_tokens(new_token_list)   
    logger.info("Add tokens: {}".format(new_token_list))      

    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer, config