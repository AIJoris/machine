#! /bin/sh
TRAIN_PATH=../machine-tasks/LookupTablesIgnoreEOS/lookup-3bit/samples/sample1/train.tsv
DEV_PATH=../machine-tasks/LookupTablesIgnoreEOS/lookup-3bit/samples/sample1/validation.tsv
TEST_PATH=../machine-tasks/LookupTablesIgnoreEOS/lookup-3bit/samples/sample1/heldout_tables.tsv

EMB_SIZE=16
H_SIZE=512
DROPOUT_ENCODER=0
DROPOUT_DECODER=0
EPOCH=100
PRINT_EVERY=200
SAVE_EVERY=200
ATTN='pre-rnn'
ATTN_METHOD='mlp'
BATCH_SIZE=1

# Swap parts
CELL='gru'
LOG_FILE=swapped_gru.log
EXPT_DIR=models/swapped/gru/
A=shared_models/guided/gru/1/
B=shared_models/baseline/gru/1

# Place encoder from A on B and re-train B
REPLACE='encoder'
python3 ../machine/swap_parts.py --replace $REPLACE --load_A $A --load_B $B \
--train $TRAIN_PATH --dev $DEV_PATH --monitor $TEST_PATH --output_dir $EXPT_DIR \
--write-logs $LOG_FILE --print_every $PRINT_EVERY --embedding_size $EMB_SIZE \
--hidden_size $H_SIZE --epoch $EPOCH --save_every $SAVE_EVERY --batch_size $BATCH_SIZE \
--dropout_p_encoder $DROPOUT_ENCODER --dropout_p_decoder $DROPOUT_DECODER \
--ignore_output_eos --teacher_forcing_ratio 0 --full_focus \
--attention_method $ATTN_METHOD --attention $ATTN --rnn_cell $CELL

# Place decoder from A on B and re-train B
REPLACE='decoder'
python3 ../machine/swap_parts.py --replace $REPLACE --load_A $A --load_B $B \
--train $TRAIN_PATH --dev $DEV_PATH --monitor $TEST_PATH --output_dir $EXPT_DIR \
--write-logs $LOG_FILE --print_every $PRINT_EVERY --embedding_size $EMB_SIZE \
--hidden_size $H_SIZE --epoch $EPOCH --save_every $SAVE_EVERY --batch_size $BATCH_SIZE \
--dropout_p_encoder $DROPOUT_ENCODER --dropout_p_decoder $DROPOUT_DECODER \
--ignore_output_eos --teacher_forcing_ratio 0 --full_focus \
--attention_method $ATTN_METHOD --attention $ATTN --rnn_cell $CELL

# Place decoder embedding from A on B and re-train B
REPLACE='decoder.embedding'
python3 ../machine/swap_parts.py --replace $REPLACE --load_A $A --load_B $B \
--train $TRAIN_PATH --dev $DEV_PATH --monitor $TEST_PATH --output_dir $EXPT_DIR \
--write-logs $LOG_FILE --print_every $PRINT_EVERY --embedding_size $EMB_SIZE \
--hidden_size $H_SIZE --epoch $EPOCH --save_every $SAVE_EVERY --batch_size $BATCH_SIZE \
--dropout_p_encoder $DROPOUT_ENCODER --dropout_p_decoder $DROPOUT_DECODER \
--ignore_output_eos --teacher_forcing_ratio 0 --full_focus \
--attention_method $ATTN_METHOD --attention $ATTN --rnn_cell $CELL

# Place encoder from B on A and re-train A
REPLACE='encoder'
python3 ../machine/swap_parts.py --reverse --replace $REPLACE --load_A $A --load_B $B \
--train $TRAIN_PATH --dev $DEV_PATH --monitor $TEST_PATH --output_dir $EXPT_DIR \
--write-logs $LOG_FILE --print_every $PRINT_EVERY --embedding_size $EMB_SIZE \
--hidden_size $H_SIZE --epoch $EPOCH --save_every $SAVE_EVERY --batch_size $BATCH_SIZE \
--dropout_p_encoder $DROPOUT_ENCODER --dropout_p_decoder $DROPOUT_DECODER \
--ignore_output_eos --teacher_forcing_ratio 0 --full_focus \
--use_attention_loss --attention_method $ATTN_METHOD --attention $ATTN --rnn_cell $CELL

# Place decoder from B on A and re-train A
REPLACE='decoder'
python3 ../machine/swap_parts.py --reverse --replace $REPLACE --load_A $A --load_B $B \
--train $TRAIN_PATH --dev $DEV_PATH --monitor $TEST_PATH --output_dir $EXPT_DIR \
--write-logs $LOG_FILE --print_every $PRINT_EVERY --embedding_size $EMB_SIZE \
--hidden_size $H_SIZE --epoch $EPOCH --save_every $SAVE_EVERY --batch_size $BATCH_SIZE \
--dropout_p_encoder $DROPOUT_ENCODER --dropout_p_decoder $DROPOUT_DECODER \
--ignore_output_eos --teacher_forcing_ratio 0 --full_focus \
--use_attention_loss --attention_method $ATTN_METHOD --attention $ATTN --rnn_cell $CELL

# Place decoder embedding from B on A and re-train A
# REPLACE='decoder.embedding'
# python3 ../machine/swap_parts.py --reverse --replace $REPLACE --load_A $A --load_B $B \
# --train $TRAIN_PATH --dev $DEV_PATH --monitor $TEST_PATH --output_dir $EXPT_DIR \
# --write-logs $LOG_FILE --print_every $PRINT_EVERY --embedding_size $EMB_SIZE \
# --hidden_size $H_SIZE --epoch $EPOCH --save_every $SAVE_EVERY --batch_size $BATCH_SIZE \
# --dropout_p_encoder $DROPOUT_ENCODER --dropout_p_decoder $DROPOUT_DECODER \
# --ignore_output_eos --teacher_forcing_ratio 0 --full_focus \
# --use_attention_loss --attention_method $ATTN_METHOD --attention $ATTN --rnn_cell $CELL
