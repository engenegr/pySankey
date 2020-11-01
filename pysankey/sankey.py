# -*- coding: utf-8 -*-
"""
Produces simple Sankey Diagrams with matplotlib.
@author: Anneya Golob & marcomanz & pierre-sassoulas
                      .-.
                 .--.(   ).--.
      <-.  .-.-.(.->          )_  .--.
       `-`(     )-'             `)    )
         (o  o  )                `)`-'
        (      )                ,)
        ( ()  )                 )
         `---"\    ,    ,    ,/`
               `--' `--' `--'
                |  |   |   |
                |  |   |   |
                '  |   '   |
"""

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class pySankeyException(Exception):
    pass
class NullsInFrame(pySankeyException):
    pass
class LabelMismatch(pySankeyException):
    pass

def check_data_matches_labels(labels, data, side):
    if len(labels >0):
        if isinstance(data, list):
            data = set(data)
        if isinstance(data, pd.Series):
            data = set(data.unique().tolist())
        if isinstance(labels, list):
            labels = set(labels)
        if labels != data:
            msg = "\n"
            if len(labels) <= 20:
                msg = "Labels: " + ",".join(labels) +"\n"
            if len(data) < 20:
                msg += "Data: " + ",".join(data)
            raise LabelMismatch('{0} labels and data do not match.{1}'.format(side, msg))
    


def sankey(left, right, title=None, left_weight=None, right_weight=None, color_dict=None,
           left_labels=None, right_labels=None, aspect=4, right_color=False,
           fontsize=14, figure_name=None, close_plot=False, size=(6,6), font_family="serif"):
    '''
    Make Sankey Diagram showing flow from left-->right

    Inputs:
        left = NumPy array of object labels on the left of the diagram
        right = NumPy array of corresponding labels on the right of the diagram
            len(right) == len(left)
        left_weight = NumPy array of weights for each strip starting from the
            left of the diagram, if not specified 1 is assigned
        right_weight = NumPy array of weights for each strip starting from the
            right of the diagram, if not specified the corresponding left_weight
            is assigned
        color_dict = Dictionary of colors to use for each label
            {'label':'color'}
        left_labels = order of the left labels in the diagram
        right_labels = order of the right labels in the diagram
        aspect = vertical extent of the diagram in units of horizontal extent
        right_color = If true, each strip in the diagram will be be colored
                    according to its left label
    Ouput:
        None
    '''
    if left_weight is None:
        left_weight = []
    if right_weight is None:
        right_weight = []
    if left_labels is None:
        left_labels = []
    if right_labels is None:
        right_labels = []

    # Check weights
    if len(left_weight) == 0:
        left_weight = np.ones(len(left))
    if len(right_weight) == 0:
        right_weight = left_weight

	# if Series, use the values attr to avoid indexing issues
    if hasattr(left_weight, "values"):
        left_weight = left_weight.values
    if hasattr(right_weight, "values"):
        right_weight = right_weight.values
    if hasattr(left_labels, "values"):
        left_labels = left_labels.values
    if hasattr(right_labels, "values"):
        right_labels = right_labels.values


    plt.figure()
    plt.rc('text', usetex=False)
    plt.rc('font', family=font_family)

    # Create Dataframe
    if isinstance(left, pd.Series):
        left = left.reset_index(drop=True)
    if isinstance(right, pd.Series):
        right = right.reset_index(drop=True)
    df = pd.DataFrame({'left': left, 'right': right, 'left_weight': left_weight,
                       'right_weight': right_weight})
     
    if (df.left.isnull().sum() > 0) | (df.right.isnull().sum() > 0 ):
        raise NullsInFrame('Sankey graph does not support null values.')

    # Identify all labels that appear 'left' or 'right'
    allLabels = pd.Series(np.r_[df.left.unique(), df.right.unique()]).unique()

    # Identify left labels
    if len(left_labels) == 0:
        left_labels = pd.Series(df.left.unique()).unique()
    else:
        check_data_matches_labels(left_labels, df['left'], 'left')

    # Identify right labels
    if len(right_labels) == 0:
        right_labels = pd.Series(df.right.unique()).unique()
    else:
        check_data_matches_labels(left_labels, df['right'], 'right')
    # If no color_dict given, make one
    if color_dict is None:
        color_dict = {}
        pal = "hls"
        cls = sns.color_palette(pal, len(allLabels))
        for i, l in enumerate(allLabels):
            color_dict[l] = cls[i]
    else:
        missing = [label for label in allLabels if label not in color_dict.keys()]
        if missing:
            raise RuntimeError('color_dict specified but missing values: '
                                '{}'.format(','.join(missing)))

    # Determine widths of individual strips
    ns_l = defaultdict()
    ns_r = defaultdict()
    for l in left_labels:
        tmp_df_l = {}
        tmp_df_r = {}
        for l2 in right_labels:
            tmp_df_l[l2] = df[(df.left == l) & (df.right == l2)].left_weight.sum()
            tmp_df_r[l2] = df[(df.left == l) & (df.right == l2)].right_weight.sum()
        ns_l[l] = tmp_df_l
        ns_r[l] = tmp_df_r

    # Determine positions of left label patches and total widths
    widths_left = defaultdict()
    for i, l in enumerate(left_labels):
        tmp_df = {}
        tmp_df['left'] = df[df.left == l].left_weight.sum()
        if i == 0:
            tmp_df['bottom'] = 0
            tmp_df['top'] = tmp_df['left']
        else:
            tmp_df['bottom'] = widths_left[left_labels[i - 1]]['top'] + 0.02 * df.left_weight.sum()
            tmp_df['top'] = tmp_df['bottom'] + tmp_df['left']
            topEdge = tmp_df['top']
        widths_left[l] = tmp_df

    # Determine positions of right label patches and total widths
    widths_right = defaultdict()
    for i, l in enumerate(right_labels):
        tmp_df = {}
        tmp_df['right'] = df[df.right == l].right_weight.sum()
        if i == 0:
            tmp_df['bottom'] = 0
            tmp_df['top'] = tmp_df['right']
        else:
            tmp_df['bottom'] = widths_right[right_labels[i - 1]]['top'] + 0.02 * df.right_weight.sum()
            tmp_df['top'] = tmp_df['bottom'] + tmp_df['right']
            topEdge = tmp_df['top']
        widths_right[l] = tmp_df

    # Total vertical extent of diagram
    x_max = topEdge / aspect

    # Draw vertical bars on left and right of each  label's section & print label
    for l in left_labels:
        plt.fill_between(
            [-0.02 * x_max, 0],
            2 * [widths_left[l]['bottom']],
            2 * [widths_left[l]['bottom'] + widths_left[l]['left']],
            color=color_dict[l],
            alpha=0.99
        )
        plt.text(
            -0.05 * x_max,
            widths_left[l]['bottom'] + 0.5 * widths_left[l]['left'],
            l,
            {'ha': 'right', 'va': 'center'},
            fontsize=fontsize
        )
    for l in right_labels:
        plt.fill_between(
            [x_max, 1.02 * x_max], 2 * [widths_right[l]['bottom']],
            2 * [widths_right[l]['bottom'] + widths_right[l]['right']],
            color=color_dict[l],
            alpha=0.99
        )
        plt.text(
            1.05 * x_max, widths_right[l]['bottom'] + 0.5 * widths_right[l]['right'],
            l,
            {'ha': 'left', 'va': 'center'},
            fontsize=fontsize
        )

    # Plot strips
    for l in left_labels:
        for l2 in right_labels:
            lc = l
            if right_color:
                lc = l2
            if len(df[(df.left == l) & (df.right == l2)]) > 0:
                # Create array of y values for each strip, half at left value, half at right, convolve
                ys_d = np.array(50 * [widths_left[l]['bottom']] + 50 * [widths_right[l2]['bottom']])
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_u = np.array(50 * [widths_left[l]['bottom'] + ns_l[l][l2]] + 50 * [widths_right[l2]['bottom'] + ns_r[l][l2]])
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')

                # Update bottom edges at each label so next strip starts at the right place
                widths_left[l]['bottom'] += ns_l[l][l2]
                widths_right[l2]['bottom'] += ns_r[l][l2]
                plt.fill_between(
                    np.linspace(0, x_max, len(ys_d)), ys_d, ys_u, alpha=0.65,
                    color=color_dict[lc]
                )
    plt.gca().axis('off')
    plt.gcf().set_size_inches(size[0], size[1])
    
    if title: plt.title(title)
    
    if figure_name != None:
        plt.savefig("{}.png".format(figure_name), bbox_inches='tight', dpi=150)
    if close_plot:
        plt.close()
