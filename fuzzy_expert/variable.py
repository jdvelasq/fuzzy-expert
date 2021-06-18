class FuzzyVariable:
    """Creates a fuzzy variable.

    Args:
        name (string): variable name.
        universe (list, numpy.array): list of points defining the universe of the variable.
        sets (dict): dictionary where keys are the name of the sets, and the values correspond to the membership for each point of the universe.

    Returns:
        A fuzzy variable.

    """

    def __init__(self, name, universe, terms):
        self.name = name
        self.universe = universe
        self.terms = terms
