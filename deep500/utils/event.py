class Event(object):
    """ An object that implements (depending on the extended type) hook
        functions to handle and measure certain parts in Deep500.
        An event can also double as a Metric (using multiple inheritance),
        in order to obtain inputs/outputs and provide measurements.
    """
    pass
    