# Example usage

This is various seq2seq implementations using 1) several models (e.g., GRU, transformer) and 2) with / without batzzang.
 
In these examples, you can
1. learn how to use batzzang
2. experiment overheads of batzzang

## Notice

* Example codes are written based on [PyTorch official chatbot tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html).
* For now, examples codes are only for training. We would appreciate if someone could write evaluation codes.

## How to run

The working directory is here (i.e., `(project_root)/examples/`). 

### Preparing Data
1. Make `data/` directory under the working directory.
2. Download the data ZIP file from [here](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html), unzip, and put it in `data/`.

### Environment setup

1. Make an python3 environment (e.g., `$ virtualenv venv --python=python3.7; source venv/bin/activate`).
2. `$ pip install -r requirements.txt`
3. Install batzzang `$ cd ..; python setup.py install; cd examples`

### Running seq2seq model (with batzzang, training code only)

`$ python seq2seq_with_batzzang.py --model (gru/transformer/bert) (--no_teacher_forcing)`