import os
import nltk
from nltk.parse.stanford import StanfordParser

class Parser(object):
	def __init__(self):
		try:
			os.environ['STANFORD_PARSER_PATH'] = "/Users/zms/stanford-parser"
			os.environ['STANFORD_PARSER'] = "/Users/zms/stanford-parser/stanford-parser.jar"
			os.environ['STANFORD_MODELS'] = "/Users/zms/stanford-parser/stanford-parser-3.9.1-models.jar"
			self.parser = StanfordParser(model_path='edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz', java_options='-mx4000m')
		except:
			os.environ['STANFORD_PARSER_PATH'] = "/home/zhangms/stanford-parser"
			os.environ['STANFORD_PARSER'] = "/home/zhangms/stanford-parser/stanford-parser.jar"
			os.environ['STANFORD_MODELS'] = "/home/zhangms/stanford-parser/stanford-parser-3.9.1-models.jar"
			self.parser = StanfordParser(model_path='edu/stanford/nlp/models/lexparser/chinesePCFG.ser.gz', java_options='-mx4000m')

	def parse(self, words):
		tree = list(self.parser.parse(words))[0]
		return tree

def get_lca_length(location1, location2):
	i = 0
	while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
		i+=1
	return i

def get_labels_from_lca(ptree, lca_len, location):
	labels = []
	for i in range(lca_len, len(location)):
		labels.append(ptree[location[:i]].label())
	return labels

def findpath(ptree, pos1, pos2):
	'''
	Find path in parented tree from pos1 to pos2
	:param ptree: parented tree
	:param pos1:
	:param pos2:
	:return:
	'''
	location1 = ptree.leaf_treeposition(pos1)
	location2 = ptree.leaf_treeposition(pos2)

	# find length of least common ancestor (lca)
	lca_len = get_lca_length(location1, location2)

	# find path from the node1 to lca
	labels1 = get_labels_from_lca(ptree, lca_len, location1)
	# ignore the first element, because it will be counted in the second part of the path
	result = labels1[1:]
	# inverse, because we want to go from the node to least common ancestor
	result = result[::-1]

	# add path from lca to node2
	result = result + get_labels_from_lca(ptree, lca_len, location2)
	return result


if __name__ == '__main__':
	words = ["\u4e2d\u56fd", "\u8fdb\u51fa\u53e3", "\u94f6\u884c", "\u4e0e", "\u4e2d\u56fd", "\u94f6\u884c", "\u4eca\u5929", "\u7b7e\u7f72", "\u4e86", "\u300a", "\u51fa\u53e3", "\u5356\u65b9", "\u4fe1\u8d37", "\u59d4\u6258", "\u4ee3\u7406", "\u534f\u8bae", "\u300b", "\uff0c", "\u4e24", "\u884c", "\u95f4", "\u4e92\u5229", "\u4e92\u8865", "\u7684", "\u5408\u4f5c", "\u5173\u7cfb", "\u7531", "\u6b64", "\u8fdb\u5165", "\u4e86", "\u4e00", "\u4e2a", "\u65b0", "\u7684", "\u9636\u6bb5", "\u3002"]
	parser = Parser()
	ptree = parser.parse(words)
	path = findpath(ptree, 0, 7)
	print(path)
	ptree.draw()