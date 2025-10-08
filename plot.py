import matplotlib.pyplot as plt
import json


def plot_results(rouge_scores, name):
    fig, axs = plt.subplots(len(rouge_scores), 1, figsize=(10, 10))
    for i, (k,v) in enumerate(rouge_scores.items()):
        axs[i].set_title(k)
        for kk,vv in v.items():
            axs[i].scatter(vv['labels'], vv['data'], label=kk)
    fig.legend()
    fig.suptitle(name)
    fig.savefig(f'expiriments/plots/{name}.png')
        

if __name__ == '__main__':
    with open('expiriments/rouge_avgs.json', 'r') as file:
        avgs = json.load(file)

    scores = {}
    for x in avgs:
        if x['standard'] not in scores.keys():
            scores[x['standard']] = {}
        for k,v in x['results'].items():
            if k not in scores[x['standard']].keys():
                scores[x['standard']][k] = {}
            for kk,vv in v.items():
                if kk not in scores[x['standard']][k].keys():
                    scores[x['standard']][k][kk] = {'labels':[], 'data':[]}
                scores[x['standard']][k][kk]['labels'].append(x['gen_note'])
                scores[x['standard']][k][kk]['data'].append(vv)

    for k,v in scores.items():
        plot_results(v, k)
        
    
    
    