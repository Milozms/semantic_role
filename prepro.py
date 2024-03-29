import json
import numpy as np
import linecache
import pickle
import nltk
from tqdm import tqdm
from parser import Parser, findpath, is_in_the_same_constituent
# from stanfordcorenlp import StanfordCoreNLP
# nlp = StanfordCoreNLP('/Users/zms/stanford-corenlp-full-2016-10-31', lang='zh')
# need sudo on MacOS

def read(textfile, propfile, outfile):
	ftext = open(textfile, 'r')
	fprop = open(propfile, 'r')
	textlines = ftext.readlines()
	proplines = fprop.readlines()
	assert len(textlines) == len(proplines)
	dataset = []
	instance = {
		'words': [],
		'pos': [],
		'props': [],
		'len': 0
	}
	for i in range(len(textlines)):
		textline = textlines[i].strip()
		propline = proplines[i].strip()
		if len(textline) == 0:
			if instance['len'] > 0:
				dataset.append(instance)
			instance = {
				'words': [],
				'pos': [],
				'props': [],
				'len': 0
			}
			continue
		word, pos = textline.split()
		instance['words'].append(word)
		instance['pos'].append(pos)
		prop = propline.split()
		instance['props'].append(prop)
		instance['len'] += 1

	ftext.close()
	fprop.close()

	'''
	convert props format to IOB format
	'''
	for idx, instance in enumerate(tqdm(dataset)):
		verbcnt = len(instance['props'][0]) - 1
		verbs = []
		tags = [[] for i in range(verbcnt)]
		for propidx, prop in enumerate(instance['props']):
			if prop[0] != '-':
				verbs.append((propidx, prop[0]))  # (verb_postion, verb)
			for i in range(verbcnt):
				tags[i].append(prop[i+1])
		for i in range(verbcnt):
			curtag = None
			newtags = []
			for tag in tags[i]:
				if tag[0] == '(':
					curtag = tag.strip('()*')
					curtag_split = curtag.split('*')
					if len(curtag_split) > 1 and curtag_split[0] == curtag_split[1]:
						curtag = curtag_split[0]
					newtags.append(curtag + '-B')
				else:
					if curtag:
						newtags.append(curtag + '-I')
					else:
						newtags.append('O')
				if tag[-1] == ')':
					curtag = None
			tags[i] = newtags
		instance['tags'] = tags
		instance['verbs'] = verbs  # (verb_postion, verb)
		instance['ner'] = get_ner(instance['words'])
		instance.pop('props')
		dataset[idx] = instance

	# with open(outfile + '.show', 'w') as outf:
	# 	for instance in dataset:
	# 		outf.write('\t'.join(instance['words']) + '\n')
	# 		outf.write('\t'.join(instance['pos']) + '\n')
	# 		outf.write('\t'.join(instance['verbs']) + '\n')
	# 		for tags in instance['tags']:
	# 			outf.write('\t'.join(tags) + '\n')
	# 		outf.write('\n')

	with open(outfile + '.ner.json', 'w') as outf:
		json.dump(dataset, outf)
	return dataset

def readtest(textfile, propfile, outfile):
	ftext = open(textfile, 'r')
	fprop = open(propfile, 'r')
	textlines = ftext.readlines()
	proplines = fprop.readlines()
	assert len(textlines) == len(proplines)
	dataset = []
	instance = {
		'words': [],
		'pos': [],
		'props': [],
		'len': 0
	}
	for i in range(len(textlines)):
		textline = textlines[i].strip()
		propline = proplines[i].strip()
		if len(textline) == 0:
			if instance['len'] > 0:
				dataset.append(instance)
			instance = {
				'words': [],
				'pos': [],
				'props': [],
				'len': 0
			}
			continue
		word, pos = textline.split()
		instance['words'].append(word)
		instance['pos'].append(pos)
		prop = propline.split()
		instance['props'].append(prop)
		instance['len'] += 1

	ftext.close()
	fprop.close()

	'''
	convert props format to IOB format
	'''
	for idx, instance in enumerate(tqdm(dataset)):
		verbcnt = len(instance['props'][0]) - 1
		verbs = []
		for propidx, prop in enumerate(instance['props']):
			if prop[0] != '-':
				verbs.append((propidx, prop[0]))  # (verb_postion, verb)
		assert verbcnt == len(verbs)
		instance['verbs'] = verbs  # (verb_postion, verb)
		instance['ner'] = get_ner(instance['words'])
		instance.pop('props')
		dataset[idx] = instance

	with open(outfile + '.ner.json', 'w') as outf:
		json.dump(dataset, outf)
	return dataset

def get_ner(words):
	sentence = ' '.join(words)
	word_with_ner = nlp.ner(sentence)
	ner = [tag for word, tag in word_with_ner]

	return ner


def build_word_list():
	test = readtest('./data/test/test.text', './data/test/test.prop.noanswer', './data/test/test')
	dev = read('./data/dev/dev.text', './data/dev/dev.props', './data/dev/dev')
	trn = read('./data/trn/trn.text', './data/trn/trn.props', './data/trn/trn')
	wordset = set()
	for dset in [test, dev, trn]:
		for instance in dset:
			for word in instance['words']:
				wordset.add(word)
	print("Word count: %d" % len(wordset))
	wordlist = list(wordset)
	with open('./data/dicts/wordlist.json', 'w') as f:
		json.dump(list(wordlist), f)
	return wordlist

def build_word_dict_emb():
	dim = 300
	with open('./data/dicts/wordlist.json', 'r') as f:
		wordlist = json.load(f)
	word2id = {}
	for i, word in enumerate(wordlist):
		word2id[word] = i
	vocab_size = len(wordlist)
	emb = np.zeros([vocab_size, dim])
	# initialized = {}
	pretrained = 0
	# avg_sigma = 0
	# avg_mu = 0
	for line in tqdm(linecache.getlines('/Users/zms/Documents/学习资料/NLP/中文词向量embedding/vectorsw300l20.all')):
		line = line.strip()
		tokens = line.split()
		word = tokens[0]
		if word in word2id:
			vec = np.array([float(tok) for tok in tokens[-dim:]])
			wordid = word2id[word]
			emb[wordid] = vec
			# initialized[word] = True
			pretrained += 1
			# mu = vec.mean()
			# sigma = np.std(vec)
			# avg_mu += mu
			# avg_sigma += sigma
	# avg_sigma /= 1. * pretrained
	# avg_mu /= 1. * pretrained
	# for w in word2id:
	# 	if w not in initialized:
	# 		emb[word2id[w]] = np.random.normal(avg_mu, avg_sigma, (dim,))
	print('Pretrained: %d, Total: %d' % (pretrained, vocab_size))
	with open('./data/dicts/wordemb.pickle', 'wb') as f:
		pickle.dump(emb, f)
	with open('./data/dicts/word2id.json', 'w') as f:
		json.dump(word2id, f)

def get_word_features(instance, position):
	'''
	:param instance:
	:param position: the position of current word
	:return:
	'''
	assert position < instance['len']
	words = ['<S>', '<S>'] + instance['words'] + ['<E>', '<E>']
	pos = ['<S>', '<S>'] + instance['pos'] + ['<E>', '<E>']
	ner = ['<S>', '<S>'] + instance['ner'] + ['<E>', '<E>']

	pb = position + 2  # biased position
	length = instance['len']
	features = []
	fappend = features.append

	# Word unigram, bigram, trigram
	fappend('W0=%s' % words[pb])
	fappend('W-1=%s' % words[pb - 1])
	fappend('W-2=%s' % words[pb - 2])
	fappend('W+1=%s' % words[pb + 1])
	fappend('W+2=%s' % words[pb + 2])
	fappend('W-1,0=%s,%s' % (words[pb - 1], words[pb]))
	fappend('W0,+1=%s,%s' % (words[pb], words[pb + 1]))
	fappend('W-1,0,+1=%s,%s,%s' % (words[pb - 1], words[pb], words[pb + 1]))
	fappend('W-2,-1,0=%s,%s,%s' % (words[pb - 2], words[pb - 1], words[pb]))
	fappend('W0,+1,+2=%s,%s,%s' % (words[pb], words[pb + 1], words[pb + 2]))

	# POS unigram, bigram, trigram
	fappend('P0=%s' % pos[pb])
	fappend('P-1=%s' % pos[pb - 1])
	fappend('P-2=%s' % pos[pb - 2])
	fappend('P+1=%s' % pos[pb + 1])
	fappend('P+2=%s' % pos[pb + 2])
	fappend('P-1,0=%s,%s' % (pos[pb - 1], pos[pb]))
	fappend('P0,+1=%s,%s' % (pos[pb], pos[pb + 1]))
	fappend('P-1,0,+1=%s,%s,%s' % (pos[pb - 1], pos[pb], pos[pb + 1]))
	fappend('P-2,-1,0=%s,%s,%s' % (pos[pb - 2], pos[pb - 1], pos[pb]))
	fappend('P0,+1,+2=%s,%s,%s' % (pos[pb], pos[pb + 1], pos[pb + 2]))

	# NER unigram, bigram, trigram
	fappend('N0=%s' % ner[pb])
	fappend('N-1=%s' % ner[pb - 1])
	fappend('N-2=%s' % ner[pb - 2])
	fappend('N+1=%s' % ner[pb + 1])
	fappend('N+2=%s' % ner[pb + 2])
	fappend('N-1,0=%s,%s' % (ner[pb - 1], ner[pb]))
	fappend('N0,+1=%s,%s' % (ner[pb], ner[pb + 1]))
	fappend('N-1,0,+1=%s,%s,%s' % (ner[pb - 1], ner[pb], ner[pb + 1]))
	fappend('N-2,-1,0=%s,%s,%s' % (ner[pb - 2], ner[pb - 1], ner[pb]))
	fappend('N0,+1,+2=%s,%s,%s' % (ner[pb], ner[pb + 1], ner[pb + 2]))


	ptree = instance['tree']
	loc = ptree.leaf_treeposition(position)

	# Parent category
	if len(loc) >= 1:
		parent = ptree[loc[:-1]]
		fappend('PCat=%s' % parent.label())
	# Grandparent Category
	if len(loc) >= 2:
		gparent = ptree[loc[:-2]]
		fappend('GPCat=%s' % gparent.label())

	# Neighbor word are in the same constituent
	if position >= 1:
		loc_l1 = ptree.leaf_treeposition(position - 1)
		fappend('L1ISC=%d' % is_in_the_same_constituent(loc, loc_l1))
	if position >= 2:
		loc_l2 = ptree.leaf_treeposition(position - 2)
		fappend('L2ISC=%d' % is_in_the_same_constituent(loc, loc_l2))
	if position + 1 < length:
		loc_r1 = ptree.leaf_treeposition(position + 1)
		fappend('R1ISC=%d' % is_in_the_same_constituent(loc, loc_r1))
	if position + 2 < length:
		loc_r2 = ptree.leaf_treeposition(position + 2)
		fappend('R2ISC=%d' % is_in_the_same_constituent(loc, loc_r2))

	return features

def get_relative_features(instance, verb_idx, position):
	'''
	:param instance:
	:param verb_idx: the index of verb in instance['verbs'] , also the index of the tags in instance['tags']
	:param position: the position of current word
	:return:
	'''
	assert verb_idx < len(instance['verbs'])
	verb_postion, verb = instance['verbs'][verb_idx]
	features = []
	fappend = features.append

	# Predicate relative position & distance
	if position < verb_postion:
		fappend('Before_Predicate')
	elif position == verb_postion:
		fappend('Is_Predicate')
	else:
		fappend('After_Predicate')
	fappend('Distance=%d' % (position - verb_postion))

	# path and partial path
	path, lpath, rpath = findpath(instance['tree'], position, verb_postion)
	fappend('Path=%s' % path)
	fappend('lPath=%s' % lpath)
	fappend('rPath=%s' % rpath)

	return features


def get_predicate_features(instance, verb_idx):
	'''
	:param instance:
	:param verb_idx: the index of verb in instance['verbs'] , also the index of the tags in instance['tags']
	:param position: the position of current word
	:return:
	'''
	assert verb_idx < len(instance['verbs'])
	words = ['<S>', '<S>'] + instance['words'] + ['<E>', '<E>']
	pos = ['<S>', '<S>'] + instance['pos'] + ['<E>', '<E>']
	length = instance['len']
	verb_postion, verb = instance['verbs'][verb_idx]
	vpb = verb_postion + 2  # biased position
	features = []
	fappend = features.append

	# Sentence level features
	# Length
	fappend('Len=%d' % length)

	# Predicate & POS & Number
	fappend('Pred=%s' % verb)
	fappend('PredPOS=%s' % pos[vpb])
	fappend('PredNum=%d' % len(instance['verbs']))

	# Predicate context word unigram, bigram, trigram
	fappend('PW-1,0=%s,%s' % (words[vpb - 1], words[vpb]))
	fappend('PW0,+1=%s,%s' % (words[vpb], words[vpb + 1]))
	fappend('PW-1,0,+1=%s,%s,%s' % (words[vpb - 1], words[vpb], words[vpb + 1]))
	fappend('PW-2,-1,0=%s,%s,%s' % (words[vpb - 2], words[vpb - 1], words[vpb]))
	fappend('PW0,+1,+2=%s,%s,%s' % (words[vpb], words[vpb + 1], words[vpb + 2]))

	# Predicate context POS unigram, bigram, trigram
	fappend('PP-1,0=%s,%s' % (pos[vpb - 1], pos[vpb]))
	fappend('PP0,+1=%s,%s' % (pos[vpb], pos[vpb + 1]))
	fappend('PP-1,0,+1=%s,%s,%s' % (pos[vpb - 1], pos[vpb], pos[vpb + 1]))
	fappend('PP-2,-1,0=%s,%s,%s' % (pos[vpb - 2], pos[vpb - 1], pos[vpb]))
	fappend('PP0,+1,+2=%s,%s,%s' % (pos[vpb], pos[vpb + 1], pos[vpb + 2]))

	return features

def init_features_for_instance(instance, word2id):
	'''
	Initialize instance['word_features'] and instance['pred_features'], instance['word_idx'], instance['verbs_id']
	:param instance:
	:return:
	'''
	word_features = []
	word_idx = []
	words = instance['words']
	length = len(words)
	assert length == instance['len']
	for pos in range(length):
		word_idx.append(word2id[words[pos]])
		word_features.append(get_word_features(instance, pos))
	instance['word_features'] = word_features
	instance['word_idx'] = word_idx
	pred_features = []
	vcnt = len(instance['verbs'])
	verbs_id = []
	for verb_idx in range(vcnt):
		pred_features.append(get_predicate_features(instance, verb_idx))
		verb_pos, verb = instance['verbs'][verb_idx]
		verbs_id.append(word2id[verb])
	instance['pred_features'] = pred_features
	instance['verbs_id'] = verbs_id
	return instance

def init_features_for_dset(dset, word2id):
	for instance in dset:
		init_features_for_instance(instance, word2id)
	return dset

def get_static_features(instance, verb_idx, position):
	'''
	Features except tag features: word_features,
	Before call this function, instance['word_features'] and instance['pred_features'] must be initialized!!!!
	:param instance:
	:param verb_idx:
	:param position:
	:return:
	'''
	return instance['word_features'][position] + instance['pred_features'][verb_idx] \
		   + get_relative_features(instance, verb_idx, position)

def get_tag_features(prev1_tag):
	features = []
	fappend = features.append
	fappend('T-1=' + prev1_tag)
	return features

def get_all_classes(dataset):
	classes = set()
	for instance in dataset:
		for tags in instance['tags']:
			for tag in tags:
				classes.add(tag)
	classes = list(classes)
	classes.sort()
	with open('./data/classes.json', 'w') as f:
		json.dump(classes, f)
	return classes

def load_dset(file):
	with open('./data/%s/%s.ner.json' % (file, file), 'r') as f:
		dset = json.load(f)
		return dset

def get_trees_for_dset():
	parser = Parser()
	files = ['trn', 'dev', 'test']
	for file in files:
		with open('./data/%s/%s.json' % (file, file), 'r') as f:
			dset = json.load(f)
			outfile = open('./data/%s/%s.tree' % (file, file), 'w')
			# if file == 'trn':
			# 	dset = dset[1462:]
			for instance in tqdm(dset):
				try:
					tree = parser.parse(instance['words'])
					outfile.write(tree._pformat_flat(nodesep='', parens='()', quotes=False))
				except:
					outfile.write('Parse Error')
				outfile.write('\n')
			outfile.close()

def read_trees_for_dset(file, dset):
	with open('./data/%s/%s.tree' % (file, file), 'r') as f:
		flines = f.readlines()
		for idx, instance in enumerate(dset):
			line = flines[idx]
			ptree = nltk.tree.ParentedTree.fromstring(line)
			instance['tree'] = ptree
			dset[idx] = instance
	return dset


if __name__ == '__main__':
	trn = read('./data/trn/trn.text', './data/trn/trn.props', './data/trn/trn')
	dev = readtest('./data/dev/dev.text', './data/dev/dev.props', './data/dev/dev')
	readtest('./data/test/test.text', './data/test/test.prop.noanswer', './data/test/test')
	classes = get_all_classes(dev + trn)
	build_word_list()
	build_word_dict_emb()
	get_trees_for_dset()