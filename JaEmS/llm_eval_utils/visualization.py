from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil
import numpy as np
import pandas as pd


# https://seaborn.pydata.org/examples/horizontal_boxplot.html
def overlayed_box(data, x, y, hue=None):
    sns.set_theme(style="ticks")
    
    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(7, 6))
    #ax.set_xscale("log")
    
    # Plot the orbital period with horizontal boxes
    sns.boxplot(
        data, x=x, y=y, hue=hue,
        whis=[0, 100], width=.6, palette="viridis"
    )
    
    # Add in points to show each observation
    sns.stripplot(data, x=x, y=y, size=4, color="0.1")
    
    # Tweak the visual presentation
    ax.xaxis.grid(True)
    #ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    return f, ax


def grid_heatmap(data, x, y, values, grid_cols=['test_samples'],
                 threshold=1, show_percentages=False, filter_empty=True, 
                 rotate_labs=True, kwargs={}, shared_value_scale=True, 
                 lower=-np.inf, ncols=5, width=10, subplot_title='Topic: {}'):
    """
    compute all of the samples in the categories and divide the
    the counts with values lower than the threshold
    grid = cols, rows (if thats not one column
    x, y
    """
    if filter_empty:
        data = data[~(data[x] == '-').any(axis=1)]
        
    counts = compute_threshold_perc(data, x + y + grid_cols, values, threshold, lower)
    counts['counts'] = counts[values]
    
    if not show_percentages:
        full = compute_threshold_counts(data, x + y + grid_cols, values, 1)
        counts['counts'] = counts[values]*full[values]
        kwargs = dict(fmt=".0f", **kwargs)
    else:
        kwargs = dict(fmt=".2f", **kwargs)
    counts_table = counts.pivot(index=grid_cols + x, columns=y, values='counts')
    counts_table_pp = counts_table.fillna(0).reset_index() 
    
    rows = list(sorted(counts[grid_cols[0]].unique()))
    nrows = ceil(len(rows)/ncols)
    
    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(12, 5*nrows)
    )
    htmp_kwargs = dict(vmax=counts['counts'].max() if shared_value_scale else None,
                       vmin=0, cmap='viridis', cbar=not shared_value_scale)
    xvals = list(counts[x[0]].unique())
    empty = pd.Series(len(counts_table.columns))
    for r, r_lab in enumerate(rows):
        plot_table = counts_table.loc[r_lab]
        # check for missing counts for group from x and add an empty row
        for xval in xvals:
            if xval not in plot_table.index:
                plot_table.loc[xval] = np.nan
        if len(axs.shape) > 1:                 
            curr_ax = axs[r // ncols][r % ncols]
        else:
            curr_ax = axs[r]
            
        if r == len(rows) - 1:
            htmp_kwargs['cbar'] = True
        plot_table.sort_index(inplace=True)
        sns.heatmap(plot_table, ax=curr_ax, **htmp_kwargs)
        
        if r % ncols != 0:
            curr_ax.set_yticks([], labels=[])
            curr_ax.set(ylabel='')
        if r < len(rows) - ncols:
            curr_ax.set_xticks([], labels=[])
            curr_ax.set(xlabel='')
        
        curr_ax.set(title=subplot_title.format(r_lab))   

    for r in range(len(rows), nrows*ncols):
        axs[-1][r % 4].set_visible(False)
        
    return fig, axs


def compute_threshold_perc(data, cols, values, threshold, 
                           lower_thresh=-np.inf):
    """
    describe inputs: lalala
    describe function: output counts of values in val_cols under threshold,
                       for group in label_col 
    """
    filtered_data = data.copy()
    filtered_data[values] = ((data[values] <= threshold) & 
                             
                             (data[values] >= lower_thresh))
    counts = (filtered_data
              .groupby(cols)[values]
              .mean().reset_index())
    return counts    


def compute_threshold_counts(data, cols, values, threshold,
                             lower_thresh=-np.inf):
    """
    describe inputs: lalala
    describe function: output counts of values in val_cols under threshold,
                       for group in label_col 
    """
    filtered_data = data[data[values] <= threshold]
    counts = (filtered_data
              .groupby(cols + [values])
              .count().reset_index())
    return counts    


def count_heatmap(data, x, y, values, threshold=1, 
                  show_percentages=False, filter_empty=True, 
                 rotate_labs=True, kwargs={}):
    if filter_empty:
        data = data[~(data[x + y] == '-').any(axis=1)]
        
    counts = compute_threshold_counts(data, x + y, values, threshold)
    counts = counts.pivot(index=x, columns=y, values=values)
    
    if show_percentages:
        full = compute_threshold_counts(data, x + y, values, 1)
        full = full.pivot(index=x, columns=y, values=values)
        counts /= full
        kwargs = dict(fmt="0.2f", **kwargs)
    else:
        kwargs = dict(fmt=".0f", **kwargs)
    counts = counts.fillna(0)
    rows, cols = counts.shape
    width = 1 if kwargs['annot'] else .5
    fig, ax = plt.subplots(figsize=(cols*width, rows/4))
    
    sns.heatmap(counts, **kwargs)
    # https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    if rotate_labs:
        ax.set_xticks(range(cols), labels=counts.columns,
                  rotation=-30, ha="right", rotation_mode="anchor")
        ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
        ax.set(title=' '.join(y), xlabel='')
    return fig, ax  

def display_outliers(data, x, y, values, threshold=1, 
                     show_percentages=False, filter_empty=True):
    kwargs = dict(annot=True, cmap='viridis')
    return count_heatmap(data, x, y, values, threshold, show_percentages,
                  filter_empty, kwargs=kwargs)
    

def overview_outliers(data, x, y, values, threshold=1, 
                     show_percentages=False, filter_empty=True):
    kwargs = dict(annot=False, cmap='viridis')
    return count_heatmap(data, x, y, values, threshold, show_percentages,
                  filter_empty, kwargs=kwargs)


def ridgeline(df, hue, plot, sort_vals):
    darkgreen  = '#9BC184'
    midgreen   = '#C2D6A4'
    lightgreen = '#E7E5CB'
    darkgrey   = '#525252'

    colors = [lightgreen, midgreen, darkgreen, midgreen, lightgreen]
    
    # iterate over axes
    words = df.sort_values(sort_vals)[hue].unique().tolist()
    
    fig, axs = plt.subplots(nrows=len(words), ncols=1, figsize=(6, len(words)//2))
    axs = axs.flatten() # needed to access each individual axis
    
    for i, word in enumerate(words):
    
        # subset the data for each word
        subset = df[df[hue] == word]
    
        # plot the distribution of prices
        sns.kdeplot(
            subset[plot],
            fill=True,
            #bw_adjust = bandwidth,
            ax=axs[i],
            color='grey',
            edgecolor='lightgrey'
        )
    
        # global mean reference line
        # global_mean = rent[plot].mean()
        # axs[i].axvline(global_mean, color='#525252', linestyle='--')
    
        # compute quantiles
        quantiles = np.percentile(subset[plot], [2.5, 10, 25, 75, 90, 97.5])
        quantiles = quantiles.tolist()
    
        # fill space between each pair of quantiles
        for j in range(len(quantiles) - 1):
            axs[i].fill_between(
                [quantiles[j], # lower bound
                 quantiles[j+1]], # upper bound
                0, # max y=0
                0.2, # max y=0.0002
                color=colors[j]
            )
           # display word on left
        axs[i].text(
            0.01, 0,
            str(word), #.upper(),
            ha='left',
            fontsize=10,
            # fontproperties=fira_sans_semibold,
            color=darkgrey
        )
        # mean value as a reference
        mean = subset[plot].mean()
        axs[i].scatter([mean], [0.001], color='black', s=10)
    
        # set title and labels
        axs[i].set_xlim(0, 1)
        #axs[i].set_ylim(0)
        #axs[i].set_ylabel('')
    
        # remove axis
        axs[i].set_axis_off()
    
    text = f'Similarity scores for {hue} individually'
    fig.text(
        0.35, 0.88,
        text,
        ha='center',
        fontsize=10
    )
    axs[0].plot([-1, -1], [1, 1], 'red', linewidth=10)
    plt.show()