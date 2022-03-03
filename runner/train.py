import os
import logging
import torch.nn as nn

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        cfg=None,
        data_loader=None,
        model=None,
        optimizer=None,
        scheduler=None,
        convert_fn=None,
    ):

        self.cfg = cfg
        self.data_loader = data_loader
        self.data_iterator = iter(self.data_loader)
        self.model = model

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.convert_fn = convert_fn
        self._init_metric()


    def _init_metric(self):
        self.metric = {
            "global_steps": 0,
            "smooth_loss": 0.0,
        }


    def convert_batch_to_inputs(self, batch):
        return self.convert_fn(batch)


    def train_one_step(self):
        self.model.train()
        try:
            batch = next(self.data_iterator)
        except StopIteration:
            self.data_iterator = iter(self.data_loader)
            batch = next(self.data_iterator)

        inputs = self.convert_batch_to_inputs(batch)
        loss, _= self.model(**inputs)

        if self.cfg.gradient_accumulation_steps > 1:
            loss = loss / self.cfg.gradient_accumulation_steps
        loss.backward()

        if self.cfg.max_grad_norm != 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
        
        self.metric['smooth_loss'] += loss.item()/self.cfg.logging_steps
        if (self.metric['global_steps']+1)%self.cfg.gradient_accumulation_steps==0:
            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()
            self.metric['global_steps'] += 1

        if self.metric['global_steps'] % self.cfg.logging_steps == 0:
            logger.info("-----------------------global_step: {} -------------------------------- ".format(self.metric['global_steps']))
            logger.info('lr: {}'.format(self.scheduler.get_lr()[0]))
            logger.info('smooth_loss: {}'.format(self.metric['smooth_loss']))
            self.metric['smooth_loss'] = 0.0
