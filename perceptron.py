from collections import defaultdict
from prepro import extract_feature
import random
import json
import numpy as np

class Perceptron(object):
	def __init__(self):
		# Each feature gets its own weight vector, so weights is a dict-of-dicts
		# self.weights[feature][class]
		self.weights = {}
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
		self.transition = np.zeros(shape=[n_class, n_class, n_class], dtype=np.float32)

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
						features = extract_feature(instance, verb_idx, position)
						pred = self.predict(features)
						golden = instance['tags'][verb_idx][position]
						self.update_weights(golden, pred, features)
			random.shuffle(dataset)


	def viterbi_decode(self, instance, verb_idx):
		length = instance['len']
		n_class = self.n_class
		lattice = np.zeros(shape=[length][n_class][n_class], dtype=np.float32)
		back = np.zeros(shape=[length][n_class][n_class], dtype=np.int32)
		'''
		Compute scores for first postion
		'''
		position = 0
		for label_id in range(n_class):
			features = extract_feature(instance, verb_idx, position)
			feat_score = self.feature_score(features, label_id)
			trans_score = self.transition[0][1][label_id]
			lattice[position][label_id] = feat_score

		'''
		Compute scores for second postion
		'''

		'''
		Dynamic programming
		'''
		for position in range(length):
			for label_id in range(n_class):

