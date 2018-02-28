import numpy as np

baseUrl = 'http://202.120.40.109:29200'

searchAllUrl = baseUrl + '/conf/acm/_search'

ids = list(eval(open('ids_20.txt', 'r', encoding='utf-8').read()))
vocab = list(eval(open('vocab_20.txt', 'r', encoding='utf-8').read()))
topic_word = np.loadtxt("topic_word_20.txt")
doc_topics = np.loadtxt("doc_topic_20.txt")
doc_topic = []
for doc in doc_topics:
    doc_topic.append(doc.argmax())

