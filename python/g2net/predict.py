import torch
from .filter import FilterModel


def BaselineModel():
    return


def read_batch():
    """
    Reads in a batch to be predicted and converts it to a torch Tensor
    """
    return


def predict():
    batch = read_batch()
    # Filters out the negatives
    # FilterModel should ideally have a very low false positive rate
    first_pred = FilterModel(batch)
    # Argmax to get the indices of positive predictions
    isGWIdx = torch.argmax(first_pred)
    finalPosPred = BaselineModel(batch[isGWIdx])

    # Creating final prediction by including the final positive predictions
    # into first_pred
    for finalPosIdx, idx in enumerate(isGWIdx):
        first_pred[idx] = finalPosPred[finalPosIdx]

    return first_pred