from .BendingEnergy import (BendingEnergyMetric, RBFBendingEnergyLoss,
                            RBFBendingEnergyLossA)
from .CrossCorrelation import (LocalCrossCorrelation2D,
                               WeightedLocalCrossCorrelation2D,
                               NCC_vxm2D,
                               LocalCrossCorrelation2D_FF,
                               LocalCrossCorrelation2D_ROI)
from .DiceCoefficient import DiceCoefficient, DiceCoefficientAll
from .Distance import MaxMinPointDist, SurfaceDistanceFromSeg
from .Jacobian import JacobianDeterminantLoss, JacobianDeterminantMetric
from .MeanSquareError import MeanSquareError

LOSSDICT = {
    'LCC': LocalCrossCorrelation2D,
    'WLCC': WeightedLocalCrossCorrelation2D,
    'MSE': MeanSquareError,
    'NCC_vxm': NCC_vxm2D,
    'LCC_FF': LocalCrossCorrelation2D_FF,
    'LCC_roi': LocalCrossCorrelation2D_ROI
}
