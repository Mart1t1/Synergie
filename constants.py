from enum import Enum

treshold = -0.2

JUMPWINDOWFRAMEBEGIN = 150  # in frames
JUMPWINDOWFRAMEEND = 250  # in frames

SYNCHROFRAME = 200  # in frames

NB_CLASSES_USED = 7 # excludes false positive and none

model_filepath = "saved_models/checkpoint"

fields_to_keep = ["Gyr_X", "Gyr_Y", "Gyr_Z", "Acc_X", "Acc_Y", "Acc_Z"]


sessions = {
        "1331": {
            "path": "data/raw/2009/1331",
            "sample_time_fine_synchro": 969369596 - 4000000
        },
        "1414": {
            "path": "data/raw/2009/1414",
            "sample_time_fine_synchro": 3572382138 + ((320 - 66 + 10) * 1200000)
        },
        "1304": {
            "path": "data/raw/1110/1304",
            "sample_time_fine_synchro": 115376653
        },
        "1404": {
            "path": "data/raw/1110/1404",
            "sample_time_fine_synchro": 3702624824
        },
        "1128": {
            "path": "data/raw/1128",
            "sample_time_fine_synchro": 1479527966
        }
    }

class jumpType(Enum):
    """
    figure skating jump type enum
    """

    # toe jumps
    TOE_LOOP = 0
    FLIP = 1
    LUTZ = 2

    # edge jumps
    SALCHOW = 3
    LOOP = 4
    AXEL = 5

    # other
    FALL = 6

    FALSE_POSITIVE = 7
    NONE = 8  # none is intended to be used when the annotation could not be completed (ice skater off frame at the time of the jump)
