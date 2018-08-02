from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from plot_conf_matrix import plot_conf_matrix
import matplotlib.pyplot as plt
import numpy as np

"""
calc_error_n_plot
 computes the the classifi report, confusion matrix and plots the results
 given the Y, Y_pred and a string (label) that indicates if it's train or test
"""
def calc_error_n_plot(Y,Y_pred,label):
    print(classification_report(Y, Y_pred))
        
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(Y, Y_pred )
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_conf_matrix(cnf_matrix, classes=['photon','proton', 'helium','nitrogen','iron' ],
                            title='Confusion matrix '+label)
    plt.show()
    
    # Plot normalized confusion matrix
    plt.figure()
    plot_conf_matrix(cnf_matrix, classes=['photon','proton', 'helium','nitrogen','iron' ], normalize=True,
                            title='Normalized confusion matrix '+label)
    
    plt.show()
    return 0