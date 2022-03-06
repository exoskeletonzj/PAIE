import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"
if os.environ.get('DEBUG', False): print('\033[92m'+'Running code in DEBUG mode'+'\033[0m')
import os.path as osp
import logging

from transformers import AdamW, get_linear_schedule_with_warmup

from models import build_model
from processors import build_processor

from utils import set_seed
from runner.runner import Runner

logger = logging.getLogger(__name__)


def train(args, model, processor):
    set_seed(args)

    logger.info("train dataloader generation")
    train_examples, train_features, train_dataloader, args.train_invalid_num = processor.generate_dataloader('train')
    logger.info("dev dataloader generation")
    dev_examples, dev_features, dev_dataloader, args.dev_invalid_num = processor.generate_dataloader('dev')
    logger.info("test dataloader generation")
    test_examples, test_features, test_dataloader, args.test_invalid_num = processor.generate_dataloader('test')

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps*args.warmup_steps, num_training_steps=args.max_steps)

    runner = Runner(
        cfg=args,
        data_samples=[train_examples, dev_examples, test_examples],
        data_features=[train_features, dev_features, test_features],
        data_loaders=[train_dataloader, dev_dataloader, test_dataloader],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_fn_dict=None,
    )
    runner.run()


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
    set_seed(args)

    model, tokenizer, config = build_model(args, args.model_type) 
    model.to(args.device)

    processor = build_processor(args, tokenizer)

    logger.info("Training/evaluation parameters %s", args)
    train(args, model, processor)
            

if __name__ == "__main__":
    main()