# Data folder structure

```bash
data
  ├── ace_eeqa
  │   ├── train_convert.json
  │   ├── dev_convert.json
  │   └── test_convert.json
  ├── ace_oneie
  │   ├── train.oneie.json
  │   ├── dev.oneie.json
  │   └── test.oneie.json
  ├── RAMS_1.0
  │   └── data
  │       ├── train.jsonlines
  │       ├── dev.jsonlines
  │       └── test.jsonlines
  ├── WikiEvent
  │   └── data
  │       ├── train.jsonl
  │       ├── dev.jsonl
  │       └── test.jsonl
  ├── prompts
  │   ├── prompts_ace_full.csv
  │   ├── prompts_wikievent_full.csv
  │   └── prompts_rams_full.csv
  └── dset_meta
      ├── description_ace.csv
      ├── description_rams.csv
      └── description_wikievent.csv
```

# set Pretrained Checkpoints
```bash
ckpts
  ├── bart-base
  │   └── facebook official files ...
  └── bart-large
      └── facebook official files ...
```

> In this release, we only contain ```PAIE``` standard model

