from gensim.models import word2vec
import logging
from gensim import corpora
from gensim import models
import os
import codecs
import numpy as np
import matplotlib.pyplot as plt
from pylab import mpl
import random
import heapq

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
# sentences = word2vec.Text8Corpus('all2.txt')
# model = word2vec.Word2Vec(sentences, size=100, window=5)
# model.save('all2.model')
# print(model['man'])
	

if __name__ == '__main__':
	model = word2vec.Word2Vec.load('all2.model')
	# print(model['man'])

	corpus = []
	word_list = []

	with codecs.open('all2.txt', 'rb', 'utf-8', 'ignore')as f:
		lines = f.readlines()
		for i in range(0, len(lines)):
			word_list.append(lines[i].split(' '))
	# print(word_list[:2])

	dictionary = corpora.Dictionary(word_list)
	new_corpus = [dictionary.doc2bow(text) for text in word_list]
	# print(new_corpus[:2])
	word2id = dictionary.token2id
	id2word = dict(zip(word2id.values(), word2id.keys()))
	# print(len(dictionary.token2id))
	
	tfidf = models.TfidfModel(new_corpus)
	tfidf.save('tfidf.model')
	# tfidf = models.TfidfModel.load('tfidf.model')

	tfidf_vec = []
	for i in range(0, len(word_list)):
		tfidf_vec.append(sorted(tfidf[dictionary.doc2bow(word_list[i])], key=lambda x:x[1])[::-1])
	# print(tfidf_vec[:2])

	
	
	#labels
	labels = []
	for i in range(510):
		labels.append(0)
	
	for i in range(386):
		labels.append(1)
	
	for i in range(417):
		labels.append(2)
	
	for i in range(511):
		labels.append(3)
		
	for i in range(401):
		labels.append(4)
		
	
	
	cout = []
	nums_k = [5, 10, 15, 20, 25, 30, 35, 40]
	for k in nums_k:
		word_tfidfvec = []
		for i in range(0, len(tfidf_vec)):
			top_k = tfidf_vec[i][:k]
			words = []
			quanzhis = []
			word_quanzhi = []
			for pair in top_k:
				words.append(id2word[pair[0]])
				quanzhis.append(pair[1])
			totalquanzhi = sum(quanzhis)
			quanzhis = [i/totalquanzhi for i in quanzhis]
			
			vector = [0 for iii in range(100)]
			for ii in range(0, k):
				temp = []
				if words[ii] in model:
					temp = model[words[ii]].tolist()
				else:
					temp = [0 for kkk in range(100)] 
				temp = [j*quanzhis[ii] for j in temp]
				# vector = map(lambda (a, b):a+b, zip(temp, vector))
				vector = [a+b for a, b in zip(temp, vector)]
			word_quanzhi.append(words)
			# word_quanzhi.append(quanzhis)
			word_quanzhi.append(vector)
			word_tfidfvec.append(word_quanzhi)
		
		final_ans = [0 for i in range(200)]
		
		for i in range(0, 500):
			#随机取出的新闻计算最相似的10篇
			acces = []
			id = random.randint(0, 2224)
			now_label = labels[id]
			now_word_tfidfvec = word_tfidfvec[id]
			now_words = set(now_word_tfidfvec[0])
			now_vec = now_word_tfidfvec[1]
			alphs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
			for alph in alphs:
				result = []
				for i in range(0, 2225):
					temp_words = set(word_tfidfvec[i][0])
					temp_vec = word_tfidfvec[i][1]
					ans = alph * len(now_words&temp_words)/k 
						  + (1-alph) * (np.dot(now_vec, temp_vec)/(np.linalg.norm(now_vec)*(np.linalg.norm(temp_vec)))+1)/2
					result.append(ans)
				#计算前10的id
				top_ids = heapq.nlargest(11, range(len(result)), result.__getitem__)[1:]
				count = 0
				for tid in top_ids:
					if labels[tid] == now_label:
						count += 1
				acc = count / 10
				acces.append(acc)
			final_ans = [a+b for a, b in zip(final_ans, acces)]	
		
		final_ans = [i/500 for i in final_ans]			
		cout.append(final_ans)	
		
	
	print(cout)
	
	#图形展示
	mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
	mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
	name_list = ['0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1']
	colors = ['yellow', 'blue', 'green', 'orange', 'black', 'red', 'pink', 'cyan']

	markers = ['D', '.', '>', '<', 'x', 'H', '1', '2']
	labels = ['k=5', 'k=10', 'k=15', 'k=20', 'k=25', 'k=30', 'k=35', 'k=40']
	
	for i in range(8):
		plt.plot(name_list, cout[i], marker = markers[i], label = labels[i], c = colors[i], lw = 2)
		
	plt.xlabel('alpha', size = 20)
	plt.ylabel('acc_rate', size = 20)
	
	plt.legend()
	plt.show()	
	