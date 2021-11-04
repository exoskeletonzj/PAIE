from transformers import BartConfig, BartTokenizerFast
from .paie import PAIE
from .single_prompt import BartSingleArg

MODEL_CLASSES = {
    'paie': (BartConfig, PAIE, BartTokenizerFast),
    'base': (BartConfig, BartSingleArg, BartTokenizerFast)
}

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
 
    tokenizer.add_tokens(['<t>', '</t>'])
    model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer, config