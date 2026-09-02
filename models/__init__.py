from .ovqe import OVQE
from .mdfi import MDFI
from .stff import STFF
from .tvqe import TVQE
from .enhancer import Enhancer
from .cvqe import CVQE
from .mfqev2_feature import FeaturePQFMFQEv2
from .pqf_detector import FeaturePQFDetector

from .TVQE import *
from .STFF import *
from .mdfi_1 import MDFI_1, MDFI_2
__all__ = [
    'OVQE',
    'MDFI',
    'STFF',
    'TVQE',
    'CVQE',
    'MDFI_1',
    'MDFI_2',
    'FeaturePQFMFQEv2',
    'FeaturePQFDetector',
    
]

try:
    from .tgaf import TGAF
    __all__.append('TGAF')
except ImportError:
    print('TGAF Fail')
    pass
