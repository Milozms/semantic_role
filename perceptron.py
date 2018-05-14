from collections import defaultdict
from prepro import get_word_features
import random
import json
import numpy as np
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

	def get_tag_features(self, prev1_tag_id, prev2_tag_id=None):
		features = []
		prev1_tag = self.classes[prev1_tag_id]
		features.append('T-1=' + prev1_tag)
		if prev2_tag_id:
			prev2_tag = self.classes[prev2_tag_id]
			features.append('T-2=' + prev2_tag)
			features.append('T-2,-1=' + prev2_tag + ',' + prev1_tag)
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
			score += self.weights[feat].get(label, default=0.0)
		return score

	def train(self, niter, dataset):
		for iter in range(niter):
			for instance in dataset:
				for verb_idx in range(len(instance['verbs'])):
					for position in range(instance['len']):
						features = get_word_features(instance, verb_idx, position)
						pred = self.predict(features)
						golden = instance['tags'][verb_idx][position]
						self.update_weights(golden, pred, features)
			random.shuffle(dataset)


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
		lattice = np.zeros(shape=[length][n_class][n_class], dtype=np.float32)
		back = np.zeros(shape=[length][n_class][n_class], dtype=np.int32)
		START = self.label2id['START']  # label id for START symbol = 0

		# Compute scores for first position
		position = 0
		for label_id in range(1, n_class):
			features = get_word_features(instance, verb_idx, position)
			lattice[position][START][label_id] = self.feature_score(features, label_id)

		# Compute scores for second position
		position = 1
		for label_id in range(1, n_class):
			for prev1 in range(1, n_class):
				features = get_word_features(instance, verb_idx, position) + self.get_tag_features(prev1)
				lattice[position][prev1][label_id] = \
					lattice[position - 1][START][prev1] + self.feature_score(features, label_id)

		# Dynamic programming
		for position in range(2, length):
			for label_id in range(1, n_class):
				for prev1 in range(1, n_class):
					word_features = get_word_features(instance, verb_idx, position)
					word_feat_score = self.feature_score(word_features, label_id)
					max_score = MINSCORE
					best_prev2 = None
					for prev2 in range(1, n_class):  # prev_2 can't be START
						tag_features = self.get_tag_features(prev1, prev2)
						score = lattice[position - 1][prev2][prev1] + self.feature_score(tag_features, label_id)
						if score > max_score:
							max_score = score
							best_prev2 = prev2
					lattice[position][prev1][label_id] = max_score + word_feat_score
					back[position][prev1][label_id] = best_prev2

		# find the best tag for the last two word
		best_last1 = None
		best_last2 = None
		max_score = MINSCORE
		for last1 in range(1, n_class):
			for last2 in range(1, n_class):
				if label_id[length - 1][last2]

		# Back track
		for position in range(length, 2, -1):


