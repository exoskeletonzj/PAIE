import os
import sys
sys.path.append("../")
import logging
from runner.train import Trainer
from runner.evaluate import Evaluator

logger = logging.getLogger(__name__)


class Runner:
    def __init__(
        self,
        cfg=None,
        data_samples=None,
        data_features=None,
        data_loaders=None,
        model=None,
        optimizer=None,
        scheduler=None,
        metric_fn_dict=None,
    ):

        self.cfg = cfg
        train_samples, dev_samples, test_samples = data_samples
        train_features, dev_features, test_features = data_features
        train_loader, dev_loader, test_loader = data_loaders

        self.trainer = Trainer(
            cfg=self.cfg,
            data_loader=train_loader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        self.dev_evaluator = Evaluator(
            cfg=self.cfg,
            data_loader=dev_loader,
            model=model,
            metric_fn_dict=metric_fn_dict,
            features=dev_features,
        )
        self.test_evaluator = Evaluator(
            cfg=self.cfg,
            data_loader=test_loader,
            model=model,
            metric_fn_dict=metric_fn_dict,
            features=test_features,
        )


    def run(self):
        for step in range(self.cfg.max_steps):
            self.trainer.train_one_step()

            if (step+1)%self.cfg.logging_steps == 0:
                self.trainer.write_log()

            if (step+1)%self.cfg.eval_steps==0:
                self.dev_evaluator.evaluate()
                self.test_evaluator.evaluate()


    def save_checkpoints(self):
        output_dir = os.path.join(self.cfg.output_dir, 'checkpoint')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.model.save_pretrained(output_dir)