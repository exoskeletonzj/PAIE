if [ $# == 0 ] 
then
    SEED=42
    LR=2e-5
else
    SEED=$1
    LR=$2
fi

work_path=exps/ace05_singleprompt/$SEED/$LR
mkdir -p $work_path

srun -p dsta --mpi=pmi2 --gres=gpu:1 -n1 --ntasks-per-node=1 --kill-on-bad-exit=1 -w SG-IDC1-10-51-2-70 python engine.py \
    --model_type base \
    --dataset_type ace_eeqa \
    --model_name_or_path ./ckpts/bart-base \
    --role_path ./data/dset_meta/description_ace.csv \
    --eval_steps 200  \
    --max_steps 10000 \
    --seed $SEED \
    --output_dir $work_path  \
    --learning_rate $LR \
    --max_enc_seq_lengt 180 \
    --max_dec_seq_length 20 \
    --batch_size 16 \
    --max_span_num 4 \
    --th_delta 6.0