from dedalus.public import EigenvalueProblem

from dedalus.tools import parsing

import logging
logger = logging.getLogger(__name__.split('.')[-1])

class EVP_homogeneous(EigenvalueProblem):
    '''
    Overrides the EVP to force homogenization
    '''
    def __init__(self, *args, **kwargs):
        logger.info('WARNING: Force-setting all equations and BCs to have RHS=0')
        super().__init__(*args, **kwargs)

    def add_bc(self, equation, **kwargs):
        equation = parsing.split_equation(equation)[0] + ' = 0'
        super().add_bc(equation, **kwargs)

    def add_equation(self, equation, **kwargs):
        equation = parsing.split_equation(equation)[0] + ' = 0'
        super().add_equation(equation, **kwargs)
                                                                                                                    
