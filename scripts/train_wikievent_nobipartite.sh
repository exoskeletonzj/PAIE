if [ $# == 0 ] 
then
    SEED=42
    LR=2e-5
else
    SEED=$1
    LR=$2
fi

work_path=exps/wikievent_nobipartite/$SEED/$LR
mkdir -p $work_path

srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-70 python -u engine.py \
    --model_type=paie \
    --dataset_type=wikievent \
    --model_name_or_path=facebook/bart-base \
    --role_path=./data/dset_meta/description_wikievent.csv \
    --prompt_path=./data/prompts/prompts_wikievent_full.csv \
    --seed=$SEED \
    --output_dir=$work_path \
    --learning_rate=$LR \
    --max_steps=20000 \
    --max_enc_seq_lengt 500 \
    --max_prompt_seq_length 80 \