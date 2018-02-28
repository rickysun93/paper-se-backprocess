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
