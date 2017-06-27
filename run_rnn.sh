#!/bin/bash
# export CUDA_VISIBLE_DEVICES=1
#
# set -e
#
# # Activate virtual environment
# source ~/.venv/bin/activate

# Model
export RNN_MODEL=RNN_BLSTM_ATTENTION

# Data
export DATA_LOADER='qa'
#export DATA_LOADER='wiki'

# export MIX_DATA

# Setup
# Sequence lengths
export MAX_ENC_SEQ_LENGTH=100
export MAX_DEC_SEQ_LENGTH=100
# export MAX_ENC_SEQ_LENGTH=500
# export MAX_DEC_SEQ_LENGTH=500

export EMBEDDING_SIZE=256
export NUM_UNITS=256
export BATCH_SIZE=128
export EPOCHS=1000
export NUM_CELLS=2
export BEAM_SIZE=1
export SAMPLE_TYPE='argmax'

# Use scheduled sampling during training
export SCHEDULED_SAMPLING=0

# Use LM support
export LM_SUPPORT=0
# export LM_SUPPORT=1

# Mix ratio
export ALPHA=0.25
#export ALPHA=0.50
#export ALPHA=0.75

# Keep probability
# export KEEP_PROB=0.50
# export KEEP_PROB=0.75
export KEEP_PROB=1.00

# Early stopping patience
export PATIENCE=10

export LEARNING_RATE=0.001
export SAVE_EPOCHS=100

export RANDOM_SEED=42

# Delete saved checkpoints
# export CLEAN_RUN=1
export CLEAN_RUN=0

export SWAP_MEMORY=1
export PARALLEL_ITERATIONS=$BATCH_SIZE

# Summary
export CREATE_SUMMARY=1
export UPDATE_EVERY=1

export VERBOSE=1

# Train model
time python3 train.py --max_enc_seq_length $MAX_ENC_SEQ_LENGTH \
  --max_dec_seq_length $MAX_DEC_SEQ_LENGTH --save_epochs $SAVE_EPOCHS \
  --embedding_size $EMBEDDING_SIZE --num_units $NUM_UNITS --batch_size $BATCH_SIZE \
  --epochs $EPOCHS --num_cells $NUM_CELLS --clean $CLEAN_RUN --model $RNN_MODEL \
  --data $DATA_LOADER --keep_prob $KEEP_PROB --patience $PATIENCE \
  --learning_rate $LEARNING_RATE --save_epochs $SAVE_EPOCHS \
  --verbose $VERBOSE --random_seed $RANDOM_SEED \
  --alpha $ALPHA \
  --scheduled_sampling $SCHEDULED_SAMPLING \
  --swap_memory $SWAP_MEMORY --parallel_iterations $PARALLEL_ITERATIONS \
  --create_summary $CREATE_SUMMARY --update_every $UPDATE_EVERY

# Test predictions
time python3 predict.py --max_enc_seq_length $MAX_ENC_SEQ_LENGTH \
  --max_dec_seq_length $MAX_DEC_SEQ_LENGTH --save_epochs $SAVE_EPOCHS \
  --embedding_size $EMBEDDING_SIZE --num_units $NUM_UNITS --batch_size $BATCH_SIZE \
  --epochs $EPOCHS --num_cells $NUM_CELLS --model $RNN_MODEL \
  --data $DATA_LOADER --keep_prob $KEEP_PROB --patience $PATIENCE \
  --learning_rate $LEARNING_RATE --verbose $VERBOSE \
  --random_seed $RANDOM_SEED \
  --alpha $ALPHA --beam_size $BEAM_SIZE \
  --sample_type $SAMPLE_TYPE --lm_support $LM_SUPPORT \
  --create_summary $CREATE_SUMMARY --update_every $UPDATE_EVERY

time python3 evaluate.py --max_enc_seq_length $MAX_ENC_SEQ_LENGTH \
  --max_dec_seq_length $MAX_DEC_SEQ_LENGTH --save_epochs $SAVE_EPOCHS \
  --embedding_size $EMBEDDING_SIZE --num_units $NUM_UNITS --batch_size $BATCH_SIZE \
  --epochs $EPOCHS --num_cells $NUM_CELLS --model $RNN_MODEL \
  --data $DATA_LOADER --keep_prob $KEEP_PROB --patience $PATIENCE \
  --learning_rate $LEARNING_RATE --verbose $VERBOSE \
  --random_seed $RANDOM_SEED \
  --alpha $ALPHA --beam_size $BEAM_SIZE \
  --sample_type $SAMPLE_TYPE --lm_support $LM_SUPPORT \
  --create_summary $CREATE_SUMMARY --update_every $UPDATE_EVERY
