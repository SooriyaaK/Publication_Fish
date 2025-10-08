from enum import Enum

"""
Enum of loss functions
"""
class LossFunctions(str, Enum):
    COSINE = "cosine",
    MSE_KICKTIME = "mse_kicktime"