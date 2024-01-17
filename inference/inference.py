import keras
import numpy as np

import constants
from utils.jump import Jump

inference_model = keras.models.load_model("../" + constants.model_filepath)  # awful but will keep it for now


def normalize(jumps):
    ds_concat = jumps.reshape(-1, jumps.shape[-1])

    means = np.mean(ds_concat)
    stds = np.std(ds_concat)

    return (jumps - means) / stds
def infer(ts_jumps: [Jump]):
    """

    :param ts_jumps: list of utils.Jump
    :return: None
    """

    jumps = []

    for jump in ts_jumps:
        jumps.append(jump.df[constants.fields_to_keep])

    jumps_np = np.array(jumps)
    normalize(jumps_np)

    for i in range(len(jumps_np)):
        pred_prob = inference_model(jumps_np[i].reshape(1, 400, 6))
        type_nb = np.argmax(pred_prob)
        type = constants.jumpType(type_nb).name
        ts_jumps[i].type = type
    return