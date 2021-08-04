from zope.interface import Interface


class TuningHyperparams(Interface):
    def tune(model):
        """tuning"""
