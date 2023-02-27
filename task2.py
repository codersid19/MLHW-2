import numpy as np
import matplotlib.pyplot as plt

# for the Given Decision tree
ln1 = {'+':14, '-':5}
ln2 = {'+':6, '-':7}
ln3 = {'+':2, '-':10}
ln4 = {'+':8, '-':6}
ln5 = {'+':5, '-':17}
ln6 = {'+':15, '-':5}

leaf_nodes = [ln1, ln2, ln3, ln4, ln5, ln6]

DecisionTree = {'A': {0: 'B', 1: 'C'},
                'B': {0: 'D', 1: 'E'},
                'C': {0: ln5, 1: ln6},
                'D': {0: ln1, 1: ln2},
                'E': {0: ln3, 1: ln4}}


error_rates = []
for i, leaf_node in enumerate(leaf_nodes):
    wrongly_classified = min(leaf_node, key = leaf_node.get)
    error_rates.append(leaf_node[wrongly_classified]/sum(leaf_node.values()))
    print(f'Leaf Node {i+1}: \n {wrongly_classified = }, total items = {leaf_node.values()}, number of wrongly classified = {leaf_node[wrongly_classified]}, error rate = {leaf_node[wrongly_classified]/sum(leaf_node.values()):0.02f}')

print(f'\nError rate of the decision tree: {np.average(error_rates): 0.2f}')