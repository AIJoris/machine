import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import random
import pickle
import copy

from collections import OrderedDict

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity, AttentionLoss, NLLLoss
from seq2seq.metrics import WordAccuracy, SequenceAccuracy, FinalTargetAccuracy, SymbolRewritingAccuracy
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField, AttentionField
from seq2seq.evaluator import Predictor, Evaluator
from seq2seq.util.checkpoint import Checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

parser = argparse.ArgumentParser()
parser.add_argument('--train', help='Training data')
parser.add_argument('--dev', help='Development data')
parser.add_argument('--monitor', nargs='+', default=[], help='Data to monitor during training')
parser.add_argument('--output_dir', default='../models', help='Path to model directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--epochs', type=int, help='Number of epochs', default=6)
parser.add_argument('--optim', type=str, help='Choose optimizer', choices=['adam', 'adadelta', 'adagrad', 'adamax', 'rmsprop', 'sgd'])
parser.add_argument('--max_len', type=int, help='Maximum sequence length', default=50)
parser.add_argument('--rnn_cell', help="Chose type of rnn cell", default='lstm')
parser.add_argument('--bidirectional', action='store_true', help="Flag for bidirectional encoder")
parser.add_argument('--embedding_size', type=int, help='Embedding size', default=128)
parser.add_argument('--hidden_size', type=int, help='Hidden layer size', default=128)
parser.add_argument('--n_layers', type=int, help='Number of RNN layers in both encoder and decoder', default=1)
parser.add_argument('--src_vocab', type=int, help='source vocabulary size', default=50000)
parser.add_argument('--tgt_vocab', type=int, help='target vocabulary size', default=50000)
parser.add_argument('--dropout_p_encoder', type=float, help='Dropout probability for the encoder', default=0.2)
parser.add_argument('--dropout_p_decoder', type=float, help='Dropout probability for the decoder', default=0.2)
parser.add_argument('--teacher_forcing_ratio', type=float, help='Teacher forcing ratio', default=0.2)
parser.add_argument('--attention', choices=['pre-rnn', 'post-rnn'], default=False)
parser.add_argument('--attention_method', choices=['dot', 'mlp', 'concat', 'hard'], default=None)
parser.add_argument('--use_attention_loss', action='store_true')
parser.add_argument('--scale_attention_loss', type=float, default=1.)
parser.add_argument('--xent_loss', type=float, default=1.)
parser.add_argument('--full_focus', action='store_true')
parser.add_argument('--batch_size', type=int, help='Batch size', default=32)
parser.add_argument('--eval_batch_size', type=int, help='Batch size', default=128)
parser.add_argument('--lr', type=float, help='Learning rate, recommended settings.\nrecommended settings: adam=0.001 adadelta=1.0 adamax=0.002 rmsprop=0.01 sgd=0.1', default=0.001)
parser.add_argument('--ignore_output_eos', action='store_true', help='Ignore end of sequence token during training and evaluation')

parser.add_argument('--load_checkpoint', help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--replace', help='Name of the component in model A to be replaced by component in model B. Default: encoder',default='encoder')
parser.add_argument('--load_A', help='The name of model A to load')
parser.add_argument('--load_B', help='The name of model B to load')
parser.add_argument('--reverse', help='Replaces component in A with B and re-train A instead of the other way around', action='store_true')
parser.add_argument('--save_every', type=int, help='Every how many batches the model should be saved', default=100)
parser.add_argument('--print_every', type=int, help='Every how many batches to print results', default=100)
parser.add_argument('--resume', action='store_true', help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', default='info', help='Logging level.')
parser.add_argument('--write-logs', help='Specify file to write logs to after training')
parser.add_argument('--cuda_device', default=0, type=int, help='set cuda device to use')

opt = parser.parse_args()
IGNORE_INDEX=-1
use_output_eos = not opt.ignore_output_eos

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)


if not opt.load_A or not opt.load_B:
    parser.error('load_A and load_B arguments are required')

if opt.use_attention_loss and not opt.attention:
    parser.error('Specify attention type to use attention loss')

if not opt.attention and opt.attention_method:
    parser.error("Attention method provided, but attention is not turned on")

if opt.attention and not opt.attention_method:
    parser.error("Attention turned on, but no attention method provided")

if opt.use_attention_loss and opt.attention_method == 'hard':
    parser.error("Can't use attention loss in combination with non-differentiable hard attention method.")

if torch.cuda.is_available():
        logging.info("Cuda device set to %i" % opt.cuda_device)
        torch.cuda.set_device(opt.cuda_device)

if opt.attention:
    if not opt.attention_method:
        logging.info("No attention method provided. Using DOT method.")
        opt.attention_method = 'dot'

############################################################################
# Prepare dataset
src = SourceField()
tgt = TargetField(include_eos=use_output_eos)

tabular_data_fields = [('src', src), ('tgt', tgt)]

if opt.use_attention_loss or opt.attention_method == 'hard':
    attn = AttentionField(use_vocab=False, ignore_index=IGNORE_INDEX)
    tabular_data_fields.append(('attn', attn))

max_len = opt.max_len

def len_filter(example):
    return len(example.src) <= max_len and len(example.tgt) <= max_len

# generate training and testing data
train = torchtext.data.TabularDataset(
    path=opt.train, format='tsv',
    fields=tabular_data_fields,
    filter_pred=len_filter
)

if opt.dev:
    dev = torchtext.data.TabularDataset(
        path=opt.dev, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter
    )
else:
    dev = None

monitor_data = OrderedDict()
for dataset in opt.monitor:
    m = torchtext.data.TabularDataset(
        path=dataset, format='tsv',
        fields=tabular_data_fields,
        filter_pred=len_filter)
    monitor_data[dataset] = m

#################################################################################
# prepare models
logging.info("loading model A from {}".format(opt.load_A))
logging.info("loading model B from {}".format(opt.load_B))

# Load pre-trained models
checkpoint_A = Checkpoint.load(opt.load_A)
checkpoint_B = Checkpoint.load(opt.load_B)

logging.info('Freezing weights of {}...'.format(opt.replace))

if opt.reverse:
    logging.info("re-training A with component from B")
    seq2seq = checkpoint_A.model
    seq2seq_replace = checkpoint_B.model
    opt.output_dir = "{}A-{}/".format(opt.output_dir,opt.replace)
else:
    logging.info("re-training B with component from A")
    seq2seq = checkpoint_B.model
    seq2seq_replace = checkpoint_A.model
    opt.output_dir = "{}B-{}/".format(opt.output_dir,opt.replace)

# Extract component from replacement model
if '.' in opt.replace:
    parent,child = opt.replace.split('.')
    component = copy.deepcopy(getattr(getattr(seq2seq_replace, parent),child))
else:
    component = copy.deepcopy(getattr(seq2seq_replace, opt.replace))

# Freeze weights of component
for param in component.parameters():
    param.requires_grad = False

# Place frozen component into to be re-trained model
if '.' in opt.replace:
    setattr(getattr(seq2seq, parent),child,component)
else:
    setattr(seq2seq,opt.replace,component)

# Prepare vocabulary
input_vocab = checkpoint_A.input_vocab
src.vocab = input_vocab

output_vocab = checkpoint_A.output_vocab
tgt.vocab = output_vocab
tgt.eos_id = tgt.vocab.stoi[tgt.SYM_EOS]
tgt.sos_id = tgt.vocab.stoi[tgt.SYM_SOS]

input_vocabulary = input_vocab.itos
output_vocabulary = output_vocab.itos

##############################################################################
# train model

# Prepare loss and metrics
pad = output_vocab.stoi[tgt.pad_token]
losses = [NLLLoss(ignore_index=pad)]
loss_weights = [float(opt.xent_loss)]

if opt.use_attention_loss:
    losses.append(AttentionLoss(ignore_index=IGNORE_INDEX))
    loss_weights.append(opt.scale_attention_loss)

for loss in losses:
  loss.to(device)

metrics = [WordAccuracy(ignore_index=pad), SequenceAccuracy(ignore_index=pad), FinalTargetAccuracy(ignore_index=pad, eos_id=tgt.eos_id)]

# create trainer
t = SupervisedTrainer(loss=losses, metrics=metrics,
                      loss_weights=loss_weights,
                      batch_size=opt.batch_size,
                      eval_batch_size=opt.eval_batch_size,
                      checkpoint_every=opt.save_every,
                      print_every=opt.print_every, expt_dir=opt.output_dir)

seq2seq, logs = t.train(seq2seq, train,
                  num_epochs=opt.epochs, dev_data=dev,
                  monitor_data=monitor_data,
                  optimizer=opt.optim,
                  teacher_forcing_ratio=opt.teacher_forcing_ratio,
                  learning_rate=opt.lr,
                  top_k=1)

if opt.write_logs:
    output_path = os.path.join(opt.output_dir, opt.write_logs)
    logs.write_to_file(output_path)
