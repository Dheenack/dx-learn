"""dxlearn: Genetic Algorithm Driven AutoML Framework."""

from dxlearn.dxclassifier import DXClassifier
from dxlearn.base.search_base import BaseSearch
from dxlearn.base.evolutionary_base import EvolutionarySearch
from dxlearn.evaluation.objectives import Objectives
from dxlearn.evaluation.scalarizer import BaseScalarizer, WeightedSumScalarizer

__version__ = "0.1.0"
__all__ = [
    "DXClassifier",
    "BaseSearch",
    "EvolutionarySearch",
    "Objectives",
    "BaseScalarizer",
    "WeightedSumScalarizer",
]
