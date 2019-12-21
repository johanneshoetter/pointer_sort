import numpy as np

def accuracy(p, y):
    total_accuracy = 0
    for pred, true in zip(p,y):
        accuracy = 0
        pred, true = pred.tolist(), true.tolist()
        end_pointer = np.max(true)
        true_pointers = true[:true.index(end_pointer)]
        pred_pointers = pred[:pred.index(end_pointer)]
        if len(true_pointers) == 0: # if no targets are defined (i.e. no match exists)
            if len(pred_pointers) == 0:
                accuracy += 1
        else:
            for p_p, t_p in zip(pred_pointers, true_pointers):
                if p_p == t_p:
                    accuracy += 1
            accuracy /= len(true_pointers)
        total_accuracy += accuracy
    total_accuracy /= len(y)
    return total_accuracy

def precision(p, y):
    total_precision = 0
    for pred, true in zip(p,y):
        precision = 0
        pred, true = pred.tolist(), true.tolist()
        end_pointer = np.max(true)
        true_pointers = true[:true.index(end_pointer)]
        pred_pointers = pred[:pred.index(end_pointer)]
        if len(true_pointers) == 0: # if no targets are defined (i.e. no match exists)
            if len(pred_pointers) == 0:
                precision += 1
        else:
            if len(pred_pointers) != 0: # avoid division by zero; if zero, then precision will be zero
                for p_p in pred_pointers:
                    if p_p in true_pointers:
                        precision += 1
                precision /= len(pred_pointers)
        total_precision += precision
    total_precision /= len(y)
    return total_precision


def recall(p, y):
    total_recall = 0
    for pred, true in zip(p,y):
        recall = 0
        pred, true = pred.tolist(), true.tolist()
        end_pointer = np.max(true)
        true_pointers = true[:true.index(end_pointer)]
        pred_pointers = pred[:pred.index(end_pointer)]
        if len(true_pointers) == 0: # if no targets are defined (i.e. no match exists)
            if len(pred_pointers) == 0:
                recall += 1
        else:
            for t_p in true_pointers:
                if t_p in pred_pointers:
                    recall += 1
            recall /= len(true_pointers)
        total_recall += recall
    total_recall /= len(y)
    return total_recall