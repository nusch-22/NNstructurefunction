#!/usr/bin/env python3
import pandas as pd
import json
import pickle
import argparse
import yaml, os
import matplotlib.pyplot as plt
import pprint
import numpy as np
import seaborn as sns

def build_dataframe(trials, bestid):
    """
    Frame the hyperscan result from the .pickle and .json file.
    """
    data = {}
    data['iteration'] = [t['tid'] for t in trials]
    data['loss'] = [t['result']['loss'] for t in trials]

    for p, k in enumerate(trials[0]['misc']['vals'].keys()):
        data[k] = []
        for t in trials:
            try:
                data[k].append(t['misc']['vals'][k][0])
            except:
                data[k].append(None)

    df = pd.DataFrame(data)
    bestdf = df[df['iteration'] == bestid['tid']]
    return df, bestdf

def plot_scans(df, bestdf, trials, bestid, filename):
    """
    Plot the hyperscan result.
    """
    print('plotting scan results...')
    # plot loss
    nplots = len(trials[0]['misc']['vals'].keys())+1
    f, axs = plt.subplots(1, nplots, sharey=True, figsize=(50,10))

    axs[0].scatter(df.get('iteration'), df.get('loss'))
    axs[0].set_xlabel('Iteration')
    axs[0].set_ylabel('FID')
    #axs[0].set_yscale('log')
    axs[0].scatter(bestdf.get('iteration'), bestdf.get('loss'))

    # plot features
    for p, k in enumerate(trials[0]['misc']['vals'].keys()):

        if k in ('learning_rate'):
            axs[p+1].scatter(df.get(k), df.get('loss'))
            if k in ('learning_rate'):
                axs[p+1].set_xscale('log')
                axs[p+1].set_xlim([1e-5, 1])
        else:
            sns.violinplot(df.get(k), df.get('loss'), ax=axs[p+1], palette="Set2",cut=0.0)
            sns.stripplot(df.get(k), df.get('loss'), ax=axs[p+1], color='gray', alpha=0.4)
        axs[p+1].set_xlabel(k)
        axs[p+1].scatter(bestdf.get(k), bestdf.get('loss'), color='orange')

    plt.savefig("{0}".format(filename), bbox_inches='tight')

def plot_correlations(df, filename):
    print('plotting correlations...')
    plt.figure(figsize=(20,20))
    sns.heatmap(df.corr(), mask=np.zeros_like(df.corr(), dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, vmax=1, vmin=-1, annot=True, fmt=".2f")
    plt.savefig("{0}".format(filename), bbox_inches='tight')

def plot_pairs(df, filename):
    print('plotting pairs')
    plt.figure(figsize=(50,50))
    sns.pairplot(df)
    plt.savefig("{0}".format(filename), bbox_inches='tight')

def get_model_with_loss(trials, loss):
    count = 0
    folder = "losses_model"
    os.mkdir(folder)
    for ls in trials:
        if ls['result']['loss'] <= loss:
            # Remove undesired keys
            ls['misc']['space_vals'].pop('scan', None)
            ls['misc']['space_vals'].pop('fl', None)
            ls['misc']['space_vals'].pop('input_replicas', None)
            ls['misc']['space_vals'].pop('out_replicas', None)
            ls['misc']['space_vals'].pop('save_output', None)
            ls['misc']['space_vals'].pop('verbose', None)
            with open('{0}/loss-model_{1}.yaml'.format(folder, count), 'w') as wfp:
                yaml.dump(ls['misc']['space_vals'], wfp, default_flow_style=False)
        count += 1

#----------------------------------------------------------------------
def main(args):
    """
    Load trial files and generate plots
    """
    with open(args.trials, 'rb') as f:
        if ".json" in args.trials:
            input_trials = json.load(f)
        elif ".pickle" in args.trials:
            input_trials = pickle.load(f)
        else:
            raise Exception ('The file is not in the correct format!')

    print('Filtering bad scans...')
    trials = []
    best = 10000
    bestid = -1
    for t in input_trials:
        if t['state'] == 2:
            trials.append(t)
            if t['result']['loss'] < best:
                best = t['result']['loss']
                bestid = t
    print(f'Number of good trials {len(trials)}')
    pprint.pprint(bestid)

    # compute dataframe
    df, bestdf = build_dataframe(trials, bestid)


    # Get models with particular loss values
    get_model_with_loss(trials, 0.0)

    # plot scans
    plot_scans(df, bestdf, trials, bestid, f'{args.trials}_scan.png')

    # plot correlation matrix
    plot_correlations(df, f'{args.trials}_corr.png')

    # plot pairs
    plot_pairs(df, f'{args.trials}_pairs.png')

#----------------------------------------------------------------------
if __name__ == "__main__":
    """
    Read command line arguments
    """
    parser = argparse.ArgumentParser(description='Analyse hyperopt GANPDFs.')
    parser.add_argument('trials', help='Take as input a .pickle or .json file with trials.')
    args = parser.parse_args()
    main(args)
