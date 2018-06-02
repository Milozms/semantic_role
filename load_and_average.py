from perceptron import loadmodel, average_model
from prepro import load_dset, read_trees_for_dset, init_features_for_dset
import sys
import json

if __name__ == '__main__':
	start = sys.argv[1]
	end = sys.argv[2]
	print('Average from %d to %d...' % (start, end))
	trn = load_dset('trn')
	dev = load_dset('dev')
	test = load_dset('test')
	with open('./data/dicts/word2id.json', 'r') as f:
		word2id = json.load(f)
	read_trees_for_dset('trn', trn)
	read_trees_for_dset('dev', dev)
	read_trees_for_dset('test', test)
	init_features_for_dset(trn, word2id)
	init_features_for_dset(dev, word2id)
	init_features_for_dset(test, word2id)
	average_model(start, end, dev, test)