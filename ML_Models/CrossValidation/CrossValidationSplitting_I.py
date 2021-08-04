from zope.interface import Interface


class CrossValidationSplittingI(Interface):

    def split():
        """split cross validation"""
