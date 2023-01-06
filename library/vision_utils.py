"""
Module contains functions for visualization  
"""

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt


def display_classifiers_statistics(classifiers_docs : list,
                                   test_size : list,
                                   statistics : np.ndarray,
                                   title : str = "",
                                   y_lim : list = None):
    
    for i, class_doc in enumerate(classifiers_docs):
        plt.plot(test_size, statistics[:,i], label = class_doc)

    if y_lim:
        plt.ylim([0 , 1])
    plt.xlim([np.min(test_size), np.max(test_size)])
    plt.grid(True)
    plt.ylabel("Value")
    plt.xlabel("Test size")
    plt.title(title)
    plt.legend()
    plt.tight_layout(pad=1.5)
    plt.show()


def display_classifiers_confusion_matrix(
    classifiers_docs : list,
    classes_docs : list,
    confusion_matrix : np.ndarray,
    test_size : list,
    display_test_size : list ):

    # Plot setting
    cm_size = 3 # Size each confusion matrix
    n_subplots = len(classifiers_docs) 
    n_cols = 4
    n_rows = int(np.ceil(n_subplots/n_cols))
    
    # For each `test_size`
    for i, test_size in enumerate(test_size):
        # Continue if test_size not in test_size_display
        if test_size not in display_test_size:
            continue

        # Prepare plot space
        fig, ax = plt.subplots(n_rows,n_cols)
        fig.set_size_inches(cm_size*n_cols, cm_size*n_rows)
        fig.tight_layout(pad=2)
        fig.suptitle("Test size: {}".format(test_size), y=1.01)

        # For each classifier
        for j, docs_class in enumerate(classifiers_docs):
            display_m = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix[i,j,:,:], 
                                                  display_labels=classes_docs)
            display_m.plot(ax=ax[j//n_cols, j%n_cols], values_format='.2f')
            display_m.im_.colorbar.remove()
            ax[j//n_cols, j%n_cols].set_title(docs_class)
            
        # Plot 
        plt.show()
        


