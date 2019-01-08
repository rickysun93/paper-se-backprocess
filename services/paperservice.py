import requests
import json
import numpy as np
from static import utils


def papersearch(query, region):
    params = {'query': {}}
    params['query']['query_string'] = {'query': query}
    if region != '':
        params['query']['query_string']['fields'] = region.split(',')
    params['size'] = 20
    headers = {'content-type': 'application/json'}
    response = requests.request('POST', utils.searchAllUrl, data=json.dumps(params), headers=headers)
    papers = []
    for paper in response.json()['hits']['hits']:
        papers.append(paper['_source']['paperid'])
    papers_np = np.array(papers)
    papertopic = papers_np.copy()
    for i, paper in enumerate(papers):
        papertopic[i] = utils.doc_topic[utils.ids.index(paper)]
    print(papertopic)
    newpapers = []
    for paper in papertopic:
        if paper != '-1':
            indices = np.where(papertopic == paper)
            newpapers = newpapers + indices[0].tolist()
            papertopic[indices] = '-1'
    result = {'hits': {}}
    result['hits']['total'] = response.json()['hits']['total']
    result['hits']['hits'] = []
    for paper in newpapers:
        result['hits']['hits'].append(response.json()['hits']['hits'][paper])
    return result


def paperidsearch(query):
    params = {'query': {}}
    params['query']['query_string'] = {'query': query, 'fields': ['paperid']}
    headers = {'content-type': 'application/json'}
    response = requests.request('POST', utils.searchAllUrl, data=json.dumps(params), headers=headers)

    topic = utils.doc_topic[utils.ids.index(query)]
    topicp = utils.doc_topics[utils.ids.index(query)]
    topics = []
    for i in np.argsort(topicp)[:-6:-1]:
        topics.append({
            'topic': int(i),
            'p': topicp[i]
        })
    wordsid = utils.topic_word[topic]
    words = []
    wordss = []
    for t in topics:
        words = []
        wordsid = utils.topic_word[t['topic']]
        for wid in np.argsort(wordsid)[:-21:-1]:
            words.append(utils.vocab[wid])
        wordss.append(words)
    result = {
        'paper': response.json()['hits']['hits'][0],
        'topics': topics,
        'words': wordss
    }
    return result
