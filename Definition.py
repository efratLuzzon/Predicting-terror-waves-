from enum import Enum, IntEnum


class GTD(Enum):
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
