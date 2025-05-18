from .SGD import SGD

# All functions use the assumption m=2
m = 2


def get_worst_instance(gamma, T, mu=0, L=1, **kwargs):
    """
    Function to get the worst instance of the SGD class given the input.

    :param gamma: Step-size
    :param T: Number of iterations
    :param mu: Strong convexity parameter (default is 0, corresponding to convexity)
    :param L: Lipschitz constant of the gradient (default is 1)

    :param kwargs: Optional and additional parameters
    
        :param objective: Objective to optimize. Should be one of ["bias", "variance"]. Default is "bias".
        :param additional_constraints: Additional conditions to add to the SGD class. Default: None.
                        Should be a function that takes in the SGD class object and modifies it in place.
        :param solver: Solver to use. Default is "MOSEK". Other option is "CLARABEL".

    :return: Solved SGD class object or None if an error occurs.
    """

    # Set up additional_conditions if not provided. Default: None
    # try:
    additional_constraints = kwargs.get('additional_constraints', lambda x: None)

    sgd = SGD(gamma, T, m, mu, L, **kwargs)
    if additional_constraints: additional_constraints(sgd)
    sgd.solve(**kwargs)
    return sgd
    # except Exception as e:
    #     print(f"Error: {e}")
    #     return None
