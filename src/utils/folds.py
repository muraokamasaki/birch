import json
import os
import sys


if len(sys.argv) == 2:
    collections = sys.argv[1]

    topic_path = os.path.join('data', 'topics', 'topics.{}.txt'.format(collections))

    topics = []
    with open(topic_path, 'r') as f:
        for line in f:
            tag = 'qid'
            ind = line.find('<{}>'.format(tag))
            if ind >= 0:
                end_ind = -7
                qid = str(int(line[ind + len(tag) + 2: end_ind]))
                topics.append(qid)

    fold_path = os.path.join('data', 'folds', '{}-folds.json'.format(collections))
    with open(fold_path, 'w') as g:
        json.dump([topics], g)

