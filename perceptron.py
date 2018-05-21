from collections import defaultdict
from prepro import read, readtest, get_tag_features, init_features_for_dset, get_static_features
import random
import json
import pickle
import numpy as np
from tqdm import tqdm
import cProfile
MINSCORE = -10000000.0

class Perceptron(object):
	def __init__(self):
		# Each feature gets its own weight vector, so weights is a dict-of-dicts
		# self.weights[feature][class]
		self.weights = {}

		# labels
		self.classes = []
		with open('./data/classes.json', 'r') as f:
			self.classes = json.load(f)
		cnt = 0
		self.label2id = {}
		for label in self.classes:
			self.label2id[label] = cnt
			cnt += 1
		n_class = len(self.classes)
		self.n_class = n_class

		# word2id and word embeddings
		with open('./data/dicts/word2id.json', 'r') as f:
			self.word2id = json.load(f)
		with open('./data/dicts/wordemb.pickle', 'rb') as f:
			self.emb = pickle.load(f)
		# weight matrix for word vectors
		self.vecdim = 300
		self.weight_mat = np.random.normal(loc=0, scale=1.0, size=[n_class, self.vecdim])
		# weight matrix for verb vectors
		# self.verb_weight_mat = np.random.normal(loc=0, scale=1.0, size=[n_class, self.vecdim])

	def get_tag_features_from_id(self, prev1_tag_id, prev2_tag_id=None, order=1):
		features = []
		prev1_tag = self.classes[prev1_tag_id]
		if order == 1:
			features.append('T-1=' + prev1_tag)
		elif order == 2 and prev2_tag_id:
			prev2_tag = self.classes[prev2_tag_id]
			features.append('T-2=' + prev2_tag)
			features.append('T-2,-1=' + prev2_tag + ',' + prev1_tag)
			return features
		return features

	def update_weights(self, label, features, word_id, val = 1.0, lr = 1.0):
		'''
		Update the feature weights.
		:param label
		:param features
		:param val: +1.0 for positive, -1.0 for negative
		:return:
		'''
		def upd_feat(c, f, v):
			self.weights[f][c] += v

		label_id = self.label2id[label]
		for f in features:
			self.weights.setdefault(f, defaultdict(float))
			upd_feat(label, f, lr*val)
		wordvec = self.emb[word_id]
		for idx in range(self.vecdim):
			self.weight_mat[label_id][idx] += val*lr*wordvec[idx]

	def feature_score(self, features, label_id):
		label = self.classes[label_id]
		score = 0.0
		for feat in features:
			if feat not in self.weights:
				continue
			score += self.weights[feat][label]
		return score

	def viterbi_decode_bigram(self, instance, verb_idx):
		'''
		:param instance:
		:param verb_idx:
		:return:
		lattice[k][u]: the maximum probability of any tag sequences ending with u at position k
		lattice[k][u] = max(arg: w)(lattice[k-1][w] + feature_score(word, w, u))
		u: prev1, w: prev2
		back[k][u]: argmax
		'''
		length = instance['len']
		n_class = self.n_class
		lattice = np.zeros(shape=[length, n_class], dtype=np.float32)
		back = np.zeros(shape=[length, n_class], dtype=np.int32)

		verb_pos, verb = instance['verbs'][verb_idx]
		vid = self.label2id['V-B']
		word_idx = instance['word_idx']

		verbid = instance['verbs_id'][verb_idx]
		verbvec = self.emb[verbid]

		# Compute scores for first position
		position = 0
		features = get_static_features(instance, verb_idx, position)
		wordid = word_idx[position]
		wordvec = self.emb[wordid]
		for label_id in range(0, n_class):
			if self.classes[label_id][-1] == 'I':
				lattice[position][label_id] = MINSCORE
				continue
			if position == verb_pos and label_id != vid:
				lattice[position][label_id] = MINSCORE
				back[position][label_id] = -1
				continue
			if position != verb_pos and label_id == vid:
				lattice[position][label_id] = MINSCORE
				back[position][label_id] = -1
				continue
			wvscore = np.dot(wordvec, self.weight_mat[label_id])
			lattice[position][label_id] = self.feature_score(features, label_id) + wvscore

		# Dynamic programming
		for position in range(1, length):
			static_features = get_static_features(instance, verb_idx, position)
			wordid = word_idx[position]
			wordvec = self.emb[wordid]
			for label_id in range(0, n_class):
				if position == verb_pos and label_id != vid:
					lattice[position][label_id] = MINSCORE
					back[position][label_id] = -1
					continue
				if position != verb_pos and label_id == vid:
					lattice[position][label_id] = MINSCORE
					back[position][label_id] = -1
					continue
				wvscore = np.dot(wordvec, self.weight_mat[label_id])
				static_feat_score = self.feature_score(static_features, label_id) + wvscore
				max_score = MINSCORE
				best_prev1 = 1
				for prev1 in self.prev_classes(label_id):
					prev1_features = self.get_tag_features_from_id(prev1)
					prev1_score = self.feature_score(prev1_features, label_id)
					score = lattice[position - 1][prev1] + prev1_score
					if score > max_score:
						max_score = score
						best_prev1 = prev1
				lattice[position][label_id] = max_score + static_feat_score
				back[position][label_id] = best_prev1

		# find the best tag for the last two word
		best_last1 = 1
		max_score = MINSCORE
		for last1 in range(0, n_class):
			if lattice[length - 1][last1] > max_score:
				max_score = lattice[length - 1][last1]
				best_last1 = last1

		decode_sequence = [0] * (length - 1) + [best_last1]

		# Back track
		for position in range(length - 2, -1, -1):
			u = decode_sequence[position + 1]
			if position == verb_pos and back[position + 1][u] != vid:
				decode_sequence[position] = vid
				continue
			decode_sequence[position] = back[position + 1][u]

		decode_tags = [self.classes[label_id] for label_id in decode_sequence]

		return decode_tags

	def prev_classes(self, label_id):
		label = self.classes[label_id]
		if label[-1] != 'I':
			return range(0, self.n_class)
		tag = label.split('-')[0]
		pclss = [self.label2id[tag + '-B'], label_id]
		return pclss

	def learn_from_one_instance(self, instance, verb_idx):
		golden = instance['tags'][verb_idx]
		pred = self.viterbi_decode_bigram(instance, verb_idx)
		length = instance['len']
		assert len(golden) == length
		assert len(pred) == length
		word_idx = instance['word_idx']

		for pos in range(length):
			if golden[pos] != pred[pos]:
				# word features, predicate_features, relative_features
				features = get_static_features(instance, verb_idx, pos)
				if pos >= 1:
					gold_features = features + get_tag_features(golden[pos - 1])
					pred_features = features + get_tag_features(pred[pos - 1])
				else:
					gold_features = features
					pred_features = features
				self.update_weights(golden[pos], gold_features, word_idx[pos], val=1.0)
				self.update_weights(pred[pos], pred_features, word_idx[pos], val=-1.0)


	def train(self, niter, dataset, validset):
		# for instance in dataset:
		# 	init_features_for_instance(instance)
		for iter in range(niter):
			print('Iteration %d:' % iter)
			for instance in tqdm(dataset):
				for verb_idx in range(len(instance['verbs'])):
					self.learn_from_one_instance(instance, verb_idx)
			random.shuffle(dataset)
			self.valid(validset, './output/valid%d.txt' % iter)
			self.save('./model/model%d.pkl' % iter)

	def test(self, dataset, filename):
		outf = open(filename, 'w')
		for instance in tqdm(dataset):
			# init_features_for_instance(instance)
			length = instance['len']
			verb_col = ['-'] * length
			verbs = instance['verbs']
			for pos, verb in verbs:
				verb_col[pos] = verb
			out_tags = [verb_col]
			for verb_idx in range(len(verbs)):
				decode_tags = self.viterbi_decode_bigram(instance, verb_idx)
				assert len(decode_tags) == length
				out_tags.append(self.format_convert(decode_tags))
			for row in range(length):
				for col in range(len(verbs) + 1):
					outf.write('%s\t' % out_tags[col][row])
				outf.write('\n')
			outf.write('\n')
		outf.close()


	def valid(self, dataset, filename):
		wordcnt = 0.0
		wordacc = 0.0
		verbcnt = 0.0
		verbacc = 0.0
		sentcnt = 0.0
		sentacc = 0.0
		outf = open(filename, 'w')
		for instance in tqdm(dataset):
			# init_features_for_instance(instance)
			length = instance['len']
			verb_col = ['-'] * length
			verbs = instance['verbs']
			instance_true = False
			if len(verbs) >= 0:
				sentcnt += 1.0
				instance_true = True
			for pos, verb in verbs:
				verb_col[pos] = verb
			out_tags = [verb_col]
			for verb_idx in range(len(verbs)):
				verbcnt += 1.0
				decode_tags = self.viterbi_decode_bigram(instance, verb_idx)
				golden_tags = instance['tags'][verb_idx]
				assert len(golden_tags) == length
				assert len(decode_tags) == length
				for idx in range(length):
					wordcnt += 1.0
					if decode_tags[idx] == golden_tags[idx]:
						wordacc += 1.0
				if golden_tags == decode_tags:
					verbacc += 1.0
				else:
					instance_true = False
				out_tags.append(self.format_convert(decode_tags))
			if instance_true:
				sentacc += 1.0
			for row in range(length):
				for col in range(len(verbs) + 1):
					outf.write('%s\t' % out_tags[col][row])
				outf.write('\n')
			outf.write('\n')
		print('Instance accuracy: %f' % (sentacc/sentcnt))
		print('Verb accuracy: %f' % (verbacc/verbcnt))
		print('Word accuracy: %f' % (wordacc/wordcnt))
		outf.close()

	def format_convert(self, tags):
		output = []
		for i, tag in enumerate(tags):
			if tag == 'O':
				output.append('*')
			elif tag == 'V-B':
				output.append('(V*V)')
			elif len(tag) >= 4:
				label = tag[:2]
				pos = tag[-1]
				if pos == 'B':
					if i < len(tags) - 1 and tags[i+1] == label + '-I':
						output.append('(%s*' % label)
					else:
						output.append('(%s*%s)' % (label, label))
				elif pos == 'I':
					if i < len(tags) - 1 and tags[i+1] == label + '-I':
						output.append('*')
					else:
						output.append('*%s)' % label)
				else:
					print('Error')
			else:
				print('Error')
		return output

	def save(self, filename):
		with open(filename, 'wb') as f:
			pickle.dump(self, f)

	def model_plus(self, mb):
		'''
		self += mb
		'''
		for feat in self.weights:
			wa = self.weights[feat]
			wb = mb.weights[feat]
			for label in wa:
				wa[label] += wb[label]

		for label_id in range(self.n_class):
			for idx in range(self.vecdim):
				self.weight_mat[label_id][idx] += mb.weight_mat[label_id][idx]

	def model_div(self, cnt):
		for feat in self.weights:
			wa = self.weights[feat]
			for label in wa:
				wa[label] /= cnt

		for label_id in range(self.n_class):
			for idx in range(self.vecdim):
				self.weight_mat[label_id][idx] /= cnt


def loadmodel(filename):
	with open(filename, 'rb') as f:
		model = pickle.load(f)
		return model

def average_model(dev):
	model = loadmodel('./model/model1.pkl')
	for iter in range(2, 8):
		model_iter = loadmodel('./model/model%d.pkl' % iter)
		model.model_plus(model_iter)
	model.model_div(8)
	model.save('./model/average.pkl')
	model.valid(dev, './output/aver.txt')


if __name__ == '__main__':
	# trn = read('./data/trn/trn.text', './data/trn/trn.props', './data/trn/trn')
	# dev = readtest('./data/dev/dev.text', './data/dev/dev.props', './data/dev/dev')
	test = readtest('./data/test/test.text', './data/test/test.prop.noanswer', './data/test/test')
	with open('./data/dicts/word2id.json', 'r') as f:
		word2id = json.load(f)
	# init_features_for_dset(trn, word2id)
	# init_features_for_dset(dev, word2id)
	init_features_for_dset(test, word2id)
	model = loadmodel('./model/average.pkl')
	# model = Perceptron()
	# cProfile.run('model.train(1, trn)')
	# model.test(dev, './output/dev.txt')
	model.test(test, './output/test.txt')
	# model.train(8, trn, dev)
	# average_model(dev)


# eval: PERL5LIB=./srlconll-1.1/lib/ perl ./srlconll-1.1/bin/srl-eval.pl ./data/dev/dev.props ./output/valid1.txt