import torch
import numpy as np
from prior_boxes     import jaccard


def AP(true_labels, pred_labels, true_boxes, pred_boxes, scores, class_instance, treshold=0.8):


    # -------------- PREPROCESSING -------------------------------

    true_labels = torch.concat(true_labels).numpy().astype(int)
    pred_labels = torch.concat(pred_labels).numpy().astype(int)
    true_boxes  = torch.concat(true_boxes)
    pred_boxes  = torch.concat(pred_boxes)

    scores      = torch.concat(scores).numpy()

    # ------------------------------------------------------------

    # select class
    true_class_label = (true_labels == class_instance).astype(int)
    pred_class_label = (pred_labels == class_instance).astype(int)


    # filtering by IOU 
    IOUs = jaccard(true_boxes, pred_boxes) 
    maxIOUS_values, maxIOUS_ind = IOUs.max(dim=0)
    treshold_maxIOUS_values = (maxIOUS_values > treshold).numpy()

    # select only relevant class
    relevant_class_ind = true_class_label[maxIOUS_ind].astype(bool)
    pred_class_label_selected = pred_class_label[relevant_class_ind]
    proff_class = np.ones_like(pred_class_label_selected) == pred_class_label_selected
    treshold_maxIOUS_values_selected = treshold_maxIOUS_values[relevant_class_ind]

    # create array with TP and FP
    prediction_tp_fp = treshold_maxIOUS_values_selected & proff_class


    # sort values in descending order
    indices = scores[relevant_class_ind].argsort(axis = 0).astype(int)[::-1].copy()
    prediction_tp_fp = prediction_tp_fp[indices].astype(int)

    # calculate precision - recall
    all_actual_positive = true_class_label.sum()
    recall = prediction_tp_fp.cumsum() / all_actual_positive
    precision = prediction_tp_fp.cumsum() / np.arange(1, len(prediction_tp_fp)+1)

    # smoothing precision function
    precision_max = []
    curr_max = -np.inf

    for val in  precision[::-1]:
        if val > curr_max:
            curr_max = val

        precision_max.append(curr_max)

    precision_max = precision_max[::-1]


    # calculate square under curve
    recall[1:] -= recall[:-1].copy()
    return np.sum(np.array(precision_max) * recall)



def mAP(true_labels, pred_labels, true_boxes, pred_boxes, scores, num_class=3, treshold=0.8):
    _map = []
    for i in range(num_class):
        Average_Precision = AP(true_labels, pred_labels, true_boxes, pred_boxes, scores, class_instance=i)
        if Average_Precision is not None:
            _map.append(Average_Precision)

    return np.mean(_map)



if __name__ == '__main__':

    true_labels =[torch.tensor([2, 2, 2, 2, 2, 1, 1, 2], dtype=torch.int32)]

    true_boxes = [torch.tensor([
                    [0.1445, 0.0958, 0.3688, 0.5083],
                    [0.5570, 0.0181, 0.7422, 0.2403],
                    [0.9133, 0.0500, 0.9992, 0.5250],
                    [0.8406, 0.0000, 0.9992, 0.3000],
                    [0.6000, 0.1042, 0.8688, 0.5736],
                    [0.2164, 0.4333, 0.2844, 0.4597],
                    [0.7234, 0.4528, 0.7914, 0.4861], 
                    [0.0016, 0.0000, 0.1625, 0.2514]],
                    dtype=torch.float64)]

    pred_labels, pred_boxes, scores = [torch.tensor([2]), torch.tensor([1]), torch.tensor([0])], \
        [
        torch.tensor([[-0.0036,  0.0015,  0.1612,  0.2483]]),
        torch.tensor([[0.7234, 0.4528, 0.7914, 0.4861]]),
        torch.tensor([[0, 0, 0, 0]]),
        ], [torch.tensor([0.9]), torch.tensor([0.1]), torch.tensor([0.0])]

    for i in range(3):
        Average_Precision = AP(true_labels, pred_labels, true_boxes, pred_boxes, scores, class_instance=i)
        print(f'for class={i}, {Average_Precision}')

    Meam_Average_Precision = mAP(true_labels, pred_labels, true_boxes, pred_boxes, scores)
    print(f'\n{Meam_Average_Precision=}')

    
