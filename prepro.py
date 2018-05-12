import json


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
	for idx, instance in enumerate(dataset):
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
					if len(curtag_split) > 1 and curtag_split[0]==curtag_split[1]:
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

	with open(outfile + '.json', 'w') as outf:
		json.dump(dataset, outf)

def extract_feature(instance, verb_idx, position):
	'''
	:param instance:
	:param verb_idx: the index of verb in instance['verbs'] , also the index of the tags in instance['tags']
	:param position: the position of current word
	:return:
	'''
	assert position < instance['len']
	assert verb_idx < len(instance['verbs'])
	words = ['<S>', '<S>'] + instance['words'] + ['<E>', '<E>']
	pos = ['<S>', '<S>'] + instance['pos'] + ['<E>', '<E>']
	pb = position + 2  # biased position
	length = instance['len']
	tags = instance['tags'][verb_idx]
	verb, verb_postion = instance['verb'][verb_idx]
	vpb = verb_postion + 2 # biased position
	features = []

	'''
	Word unigram, bigram, trigram
	'''
	features.append('W0=' + words[pb])
	features.append('W-1=' + words[pb - 1])
	features.append('W-2=' + words[pb - 2])
	features.append('W+1=' + words[pb + 1])
	features.append('W+2=' + words[pb + 2])
	features.append('W-1,0=' + words[pb - 1] + ',' + words[pb])
	features.append('W0,+1=' + words[pb] + ',' + words[pb + 1])
	features.append('W-1,0,+1=' + words[pb - 1] + ',' + words[pb] + ',' + words[pb + 1])
	features.append('W-2,-1,0=' + words[pb - 2] + ',' + words[pb - 1] + ',' + words[pb])
	features.append('W0,+1,+2=' + words[pb] + ',' + words[pb + 1] + ',' + words[pb + 2])
	'''
	POS unigram, bigram, trigram
	'''
	features.append('P0=' + pos[pb])
	features.append('P-1=' + pos[pb - 1])
	features.append('P-2=' + pos[pb - 2])
	features.append('P+1=' + pos[pb + 1])
	features.append('P+2=' + pos[pb + 2])
	features.append('P-1,0=' + pos[pb - 1] + ',' + pos[pb])
	features.append('P0,+1=' + pos[pb] + ',' + pos[pb + 1])
	features.append('P-1,0,+1=' + pos[pb - 1] + ',' + pos[pb] + ',' + pos[pb + 1])
	features.append('P-2,-1,0=' + pos[pb - 2] + ',' + pos[pb - 1] + ',' + pos[pb])
	features.append('P0,+1,+2=' + pos[pb] + ',' + pos[pb + 1] + ',' + pos[pb + 2])
	'''
	Predicate relative position & distance
	'''
	if position < verb_postion:
		features.append('Before_Predicate')
	elif position == verb_postion:
		features.append('Is_Predicate')
	else:
		features.append('After_Predicate')
	features.append('Distance=' + str(position - verb_postion))
	'''
	Suffix
	'''
	features.append('Suf2=' + words[pb][:2])
	features.append('Suf3=' + words[pb][:3])
	features.append('Suf4=' + words[pb][:4])
	'''
	Sentence level features
	'''
	'''
	Length
	'''
	features.append('Len=' + str(length))
	'''
	Predicate & POS & Number
	'''
	features.append('Pred=' + verb)
	features.append('PredPOS=' + pos[vpb])
	features.append('PredNum=' + str(len(instance['verbs'])))
	'''
	Predicate context word unigram, bigram, trigram
	'''
	features.append('PW0=' + words[vpb])
	features.append('PW-1,0=' + words[vpb - 1] + ',' + words[vpb])
	features.append('PW0,+1=' + words[vpb] + ',' + words[vpb + 1])
	features.append('PW-1,0,+1=' + words[vpb - 1] + ',' + words[vpb] + ',' + words[vpb + 1])
	features.append('PW-2,-1,0=' + words[vpb - 2] + ',' + words[vpb - 1] + ',' + words[vpb])
	features.append('PW0,+1,+2=' + words[vpb] + ',' + words[vpb + 1] + ',' + words[vpb + 2])
	'''
	Predicate context POS unigram, bigram, trigram
	'''
	features.append('PP0=' + pos[vpb])
	features.append('PP-1,0=' + pos[vpb - 1] + ',' + pos[vpb])
	features.append('PP0,+1=' + pos[vpb] + ',' + pos[vpb + 1])
	features.append('PP-1,0,+1=' + pos[vpb - 1] + ',' + pos[vpb] + ',' + pos[vpb + 1])
	features.append('PP-2,-1,0=' + pos[vpb - 2] + ',' + pos[vpb - 1] + ',' + pos[vpb])
	features.append('PP0,+1,+2=' + pos[vpb] + ',' + pos[vpb + 1] + ',' + pos[vpb + 2])

	return features




if __name__ == '__main__':
	read('./data/dev/dev.text', './data/dev/dev.props', './data/dev/dev')