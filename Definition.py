from enum import IntEnum, Enum


class GTD(IntEnum):
    ID_ATTACK = 0
    YEAR = 1
    MONTH = 2
    DAY = 3
    COUNTRY = 7
    SUCCESS = 26
    ATTACK_TYPE = 29
    WEAP_TYPE = 82
    NUM_KILL = 98
    NUM_WOUND = 101
    SELECTED_COUNTRY = 97
    DESCRIPTION = 18
    ORIGINAIZED = 58



    def __index__(self):
        return self.value

    def __int__(self):
        return int(self.value)


class DEEP_ANT(IntEnum):
    LOOKBACK_SIZE = 30
    EPOCH = 100
    KERNEL_CONV = 1
    KERNEL_POOL = 1
    IDX_YEAR = 1
    IDX_MONTH = 2
    IDX_DAY = 3
    FEATURE_NUM = 1

    def __index__(self):
        return self.value

    def __int__(self):
        return int(self.value)


class XGB(Enum):
    REGRESSOR = 0
    CLASSIFIER = 1
    LGBM = 'lgbm'
    XGB = 'xgb'
    MIN_CHILD_WEIGHT = 'min_child_weight'
    MAX_DEPTH = 'max_depth'
    SUBSAMPLE = 'subsample'
    COLSAMPLE = 'colsample_bytree'
    ETA = 'learning_rate'
    GAMMA = 'gamma'
    REG_ALPHA = 'reg_alpha'
    N_ESTIMATORS = 'n_estimators'
    MAX_DELTA_STEP = 'max_delta_step'
    COL_SAMPLE_BY_LEVEL = 'colsample_bylevel'
    REG_LAMBDA = 'reg_lambda'
    SCALE_POS_WEIGHT = 'scale_pos_weight'

    def __index__(self):
        return self.value

    def __int__(self):
        return int(self.value)


class COLOR(Enum):
    YELLOW = "\033[1;32m%s\033[0m"

    def __str__(self):
        return self.value
