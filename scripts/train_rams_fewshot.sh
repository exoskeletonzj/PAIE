if [ $# == 0 ] 
then
    SEED=42
    LR=2e-5
else
    SEED=$1
    LR=$2
fi

work_path=exps/rams_fewshot/$SEED/$LR
mkdir -p $work_path

srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-70 python -u engine.py \
    --model_type=paie \
    --dataset_type=rams \
    --model_name_or_path=facebook/bart-base \
    --role_path=./data/dset_meta/description_rams.csv \
    --prompt_path=./data/prompts/prompts_rams_full.csv \
    --seed=$SEED \
    --output_dir=$work_path \
    --learning_rate=$LR \
    --max_steps=10000 \
    --max_enc_seq_lengt 500 \
    --max_prompt_seq_length 50 \
    --keep_ratio $KEEP_RATIO \
    --bipartite
