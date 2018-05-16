from collections import defaultdict
from prepro import read, get_tag_features, init_features_for_instance, get_static_features
import random
import json
import numpy as np
from tqdm import tqdm
import cProfile
MINSCORE = -10000000.0

class Perceptron(object):
	def __init__(self):
		# Each feature gets its own weight vector, so weights is a dict-of-dicts
		# self.weights[feature][class]
		self.weights = {}
		self.classes = []
		with open('./data/classes.json', 'r') as f:
			self.classes = ['START'] + json.load(f)
		cnt = 0
		self.label2id = {}
		for label in self.classes:
			self.label2id[label] = cnt
			cnt += 1
		n_class = len(self.classes)
		self.n_class = n_class
		# self.transition = np.zeros(shape=[n_class, n_class, n_class], dtype=np.float32)

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

	def update_weights(self, golden, pred, features):
		'''
		Update the feature weights.
		:param golden: golden class
		:param pred: predicted class
		:param features
		:return:
		'''
		def upd_feat(c, f, v):
			self.weights[f][c] += v

		if golden == pred:
			return
		for f in features:
			self.weights.setdefault(f, defaultdict(float))
			upd_feat(golden, f, 1.0)
			upd_feat(pred, f, -1.0)


	def feature_score(self, features, label_id):
		label = self.classes[label_id]
		score = 0.0
		for feat in features:
			if feat not in self.weights:
				continue
			score += self.weights[feat][label]
		return score

	def viterbi_decode(self, instance, verb_idx):
		'''
		:param instance:
		:param verb_idx:
		:return:
		lattice[k][u][v]: the maximum probability of any tag sequences ending with u, v at position k
		lattice[k][u][v] = max(arg: w)(lattice[k-1][w][u] + feature_score(word, w, u, v))
		u: prev1, w: prev2
		back[k][u][v]: argmax
		'''
		length = instance['len']
		n_class = self.n_class
		lattice = np.zeros(shape=[length, n_class, n_class], dtype=np.float32)
		back = np.zeros(shape=[length, n_class, n_class], dtype=np.int32)
		START = self.label2id['START']  # label id for START symbol = 0

		# Compute scores for first position
		position = 0
		features = get_static_features(instance, verb_idx, position)
		for label_id in range(1, n_class):
			lattice[position][START][label_id] = self.feature_score(features, label_id)

		# Compute scores for second position
		position = 1
		static_features = get_static_features(instance, verb_idx, position)
		for label_id in range(1, n_class):
			static_score = self.feature_score(static_features, label_id)
			for prev1 in range(1, n_class):
				tag_features = self.get_tag_features_from_id(prev1)
				score = static_score + self.feature_score(tag_features, label_id)
				lattice[position][prev1][label_id] = \
					lattice[position - 1][START][prev1] + score

		# Dynamic programming
		for position in range(2, length):
			static_features = get_static_features(instance, verb_idx, position)
			for label_id in range(1, n_class):
				static_feat_score = self.feature_score(static_features, label_id)
				for prev1 in range(1, n_class):
					max_score = MINSCORE
					best_prev2 = None
					prev1_features = self.get_tag_features_from_id(prev1)
					prev1_score = self.feature_score(prev1_features, label_id)
					for prev2 in range(1, n_class):  # prev_2 can't be START
						prev2_features = self.get_tag_features_from_id(prev1, prev2, 2)
						prev2_score = self.feature_score(prev2_features, label_id)
						score = lattice[position - 1][prev2][prev1] + prev1_score + prev2_score
						if score > max_score:
							max_score = score
							best_prev2 = prev2
					lattice[position][prev1][label_id] = max_score + static_feat_score
					back[position][prev1][label_id] = best_prev2

		# find the best tag for the last two word
		best_last1 = None
		best_last2 = None
		max_score = MINSCORE
		for last1 in range(1, n_class):
			for last2 in range(1, n_class):
				if lattice[length - 1][last2][last1] > max_score:
					max_score = lattice[length - 1][last2][last1]
					best_last1, best_last2 = last1, last2

		decode_sequence = [0] * (length - 2) + [best_last2, best_last1]

		# Back track
		for position in range(length - 3, 0, -1):
			u = decode_sequence[position + 1]
			v = decode_sequence[position + 2]
			decode_sequence[position] = back[position + 2][u][v]

		decode_tags = [self.classes[label_id] for label_id in decode_sequence]

		return decode_tags

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
		START = self.label2id['START']  # label id for START symbol = 0

		# Compute scores for first position
		position = 0
		features = get_static_features(instance, verb_idx, position)
		for label_id in range(1, n_class):
			lattice[position][label_id] = self.feature_score(features, label_id)

		# Dynamic programming
		for position in range(1, length):
			static_features = get_static_features(instance, verb_idx, position)
			for label_id in range(1, n_class):
				static_feat_score = self.feature_score(static_features, label_id)
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
		for last1 in range(1, n_class):
			if lattice[length - 1][last1] > max_score:
				max_score = lattice[length - 1][last1]
				best_last1 = last1

		decode_sequence = [0] * (length - 1) + [best_last1]

		# Back track
		for position in range(length - 2, 0, -1):
			u = decode_sequence[position + 1]
			decode_sequence[position] = back[position + 1][u]

		decode_tags = [self.classes[label_id] for label_id in decode_sequence]

		return decode_tags

	def prev_classes(self, label_id):
		label = self.classes[label_id]
		if label[-1] != 'I':
			return range(1, self.n_class)
		tag = label.split('-')[0]
		pclss = [self.label2id[tag + '-B'], label_id]
		return pclss

	def learn_from_one_instance(self, instance, verb_idx):
		golden = instance['tags'][verb_idx]
		pred = self.viterbi_decode_bigram(instance, verb_idx)
		length = instance['len']
		assert len(golden) == length
		assert len(pred) == length

		for pos in range(length):
			if golden[pos] != pred[pos]:
				# word features, predicate_features, relative_features
				features = get_static_features(instance, verb_idx, pos)
				'''
				if pos >= 2:
					features += get_tag_features(golden[pos - 1], golden[pos - 2])
				elif pos == 1:
					features += get_tag_features(golden[pos - 1])
				'''
				if pos >= 1:
					features += get_tag_features(golden[pos - 1])
				self.update_weights(golden[pos], pred[pos], features)


	def train(self, niter, dataset, validset):
		for instance in dataset:
			init_features_for_instance(instance)
		for iter in range(niter):
			print('Iteration %d:' % iter)
			for instance in tqdm(dataset):
				for verb_idx in range(len(instance['verbs'])):
					self.learn_from_one_instance(instance, verb_idx)
			random.shuffle(dataset)
			model.valid(validset, './output/valid%d.txt' % iter)

	def valid(self, dataset, filename):
		wordcnt = 0.0
		wordacc = 0.0
		verbcnt = 0.0
		verbacc = 0.0
		sentcnt = 0.0
		sentacc = 0.0
		outf = open(filename, 'w')
		for instance in dataset:
			init_features_for_instance(instance)
			length = instance['len']
			sentcnt += 1.0
			instance_true = True
			verb_col = ['-'] * length
			verbs = instance['verbs']
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
				for col in range(len(verbs)):
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

if __name__ == '__main__':
	trn = read('./data/trn/trn.text', './data/trn/trn.props', './data/trn/trn')
	dev = read('./data/dev/dev.text', './data/dev/dev.props', './data/dev/dev')
	model = Perceptron()
	# cProfile.run('model.train(1, trn)')
	model.valid(dev, './data/0.txt')
	# model.train(16, trn, dev)