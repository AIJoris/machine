#! /bin/sh
TEST_PATH=../machine-tasks/LookupTablesIgnoreEOS/lookup-3bit/samples/sample1/new_compositions.tsv
EXPT_DIR=models/swapped/gru/B-encoder
# EXPT_DIR=shared_models/baseline/gru/1/

# Evaluate
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --test_data $TEST_PATH --ignore_output_eos
# python evaluate.py --checkpoint_path $EXPT_DIR --test_data $TEST_PATH --ignore_output_eos
