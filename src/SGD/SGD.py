import cvxpy as cp
import numpy as np
from .build_vectors import build_L
from .build_matrices import build_S
from ..Parameters import BagOfParameters, Parameter, ParameterList


class SGD:
    # Constants
    EPS = [0, 1e-6, 1e-5, 1e-4, 1e-3]

    def __init__(self, gamma, T, m=2, mu=0, L=1, **kwargs):
        """
        Initializes the SGD class.

        :param gamma: Step-size
        :param T: Number of iterations
        :param m: Number of functions (default is 2)
        :param mu: Strong convexity parameter (default is 0, corresponding to convexity)
        :param L: Lipschitz constant (default is 1)

        :param kwargs: Optional and additional parameters
            :param objective: Objective to optimize. Should be one of ["bias", "variance"] (default is "bias")
            :param solver: Solver to use. Default is "MOSEK". Other option is "CLARABEL".

        :return: None
        """

        # Set up the parameters
        self.gamma = gamma
        self.mu = mu
        self.L = L
        self.NB_ITS = T
        self.STRONGLY_CONVEX = mu > 0

        # Set up the indices related to support size to avoid confusion
        self.__setup_indices(m)

        # Set up the variables
        self.__setup_variables(**kwargs)

        # Set up the objective
        self.__setup_objective(**kwargs)

        # Set up final value
        self.value = None
        self.new_value = None

    def __setup_indices(self, m):
        """
        Defines the indices of the variables in the optimization problem. The variables are grouped as follows:
        - PARTIAL_CURR: Variables corresponding to f^(i)(x_k)
        - PARTIAL_STAR: Variables corresponding to f^(i)(x_*)
        - FULL_NEXT: Variables corresponding to x_{k+1}^(i) (and its function values and gradients)
        - STAR: Variable corresponding to x_* (and its function value and gradient)
        - CURR: Variable corresponding to x_k (and its function value and gradient)
        - NEXT: Variable corresponding to x_{k+1} (and its function value and gradient)

        :param m: Number of support points

        :return: None
        """

        # Number of support points
        self.NB_SUPPORT = m

        # Group of Variables
        self.PARTIAL_CURR = list(range(0, m))
        self.PARTIAL_STAR = list(range(m, 2 * m))
        self.FULL_NEXT = list(range(2 * m, 3 * m))

        # Special Variables
        self.CURR = 3 * m
        self.STAR = 3 * m + 1
        self.NEXT = 3 * m + 2

        # P = [*(g^(i)(x_k))_i, *(g^(i)(x_*))_i, *(g(x_{k+1}^(i))_i, x_k-x^*, dummy, dummy]
        # F = [*(f^(i)(x_k))_i, *(f^(i)(x_*))_i, *(f(x_{k+1}^(i))_i, f(x_k), dummy, dummy]
        self.NB_VARS = 3 * m + 3

        # Probability distribution (Uniform)
        self.PROBABILITY = 1 / m

    def __setup_variables(self, **kwargs):
        """
        Sets up the variables of the optimization problem. These are the coefficients of the Lyapunov's and the dual
        variables of the subproblems (dealt with analytically).

        :return: None
        """

        # Setup problem
        self.constraints = []

        # Variables representing the coefficients of the Lyapunov functions
        # Variables correspond to (a_k, e_k) and rho
        self.vars = [cp.Variable(2) for _ in range(self.NB_ITS + 1)]
        self.rho = cp.Variable()
        self.constraints += [var >= 0 for var in self.vars]
        self.constraints += [self.rho >= 0]

        # Dual multipliers for the constraint |g^*|=0
        self.taus = [cp.Variable() for _ in range(self.NB_ITS)]

        # Dual multipliers for the interpolation constraint between f_*^(i) and f_k^(i)
        self.lamb1s = [cp.Variable(self.NB_VARS) for _ in range(self.NB_ITS)]
        self.constraints += [lamb1 >= 0 for lamb1 in self.lamb1s]

        # Dual multipliers for the interpolation constraint between f_k^(i) and f_*^(i)
        self.lamb2s = [cp.Variable(self.NB_VARS) for _ in range(self.NB_ITS)]
        self.constraints += [lamb2 >= 0 for lamb2 in self.lamb2s]

        # Dual multipliers for the interpolation constraint between f_{k+1}^(i) and f_{k+1}^(j),
        # also accounting for f_k and f_*.
        self.lambs = [cp.Variable((self.NB_VARS, self.NB_VARS)) for _ in range(self.NB_ITS)]
        self.constraints += [lamb >= 0 for lamb in self.lambs]

    def __setup_constraints(self, **kwargs):
        """
        Adds the constraints required to define the PEP.

        :param normalization: a string which determines how the lyapunov parameters will be normalized (as the form a positively homogeneous family). Possible choices are:
            - 'a_first' : sets a0 to 1 and aN to 0 (DEFAULT)
            - 'rho' : sets rho to 1

        :return: None
        """

        normalization = kwargs.get('normalization', 'a_first')

        # Setup problem
        for it in range(self.NB_ITS):
            S = build_S(self, self.taus[it], self.lamb1s[it], self.lamb2s[it], self.lambs[it],
                        self.vars[it + 1], self.vars[it])
            L = build_L(self, self.lamb1s[it], self.lamb2s[it], self.lambs[it], self.rho)
            self.constraints += [S >> 0, L == 0]

        # Anchoring the Lyapunov functions
        a0, e0 = self.vars[0]
        aN, eN = self.vars[-1]
        self.constraints += [eN == 0]
        self.constraints += [aN == 1] if self.STRONGLY_CONVEX else [a0 == 1]

    def __setup_objective(self, **kwargs):
        """
        Sets the objective of the PEP. The possible objectives are:
        - "d": maximize the d term (in front of the sum) in the Lyapunov function (for convex)
        - "bias" : minimize uniquely the bias term, which with our notations is a0/d.
        - "a": maximize the a term (in front of |x_N-x^*|^2) in the Lyapunov function (for strongly convex)
        - "e_sum": minimizes the sum of (e_k), by fixing d to be its maximal value (up to a factor 1-epsilon)
        - "e_constant": e_k=e for all k and minimizes e, by fixing d to be its maximal value (up to a factor 1-epsilon)

        :param objective: Objective to optimize. Should be one of ["d", "e_sum", "e_first", "e_last", "e_constant"]

        :return: None
        """

        # Set up the objective if not provided
        objective = kwargs.get('objective', 'bias')

        # sets this as a default value because 1) it is None most of the time 2) it is weird edge case 3) anytime someone adds a new objective they are going to forget about it and it will create bugs for nothing.
        # it can be overwritten if needed when needed
        self.new_objective = None

        match objective:
            # Minimizes the bias term
            case "bias":
                self.objective = cp.Minimize(self.vars[0][0]) if self.STRONGLY_CONVEX else cp.Maximize(self.rho)

            # Minimizes the variance term subject to having found the bias term
            case "variance":
                self.objective = cp.Minimize(self.vars[0][0]) if self.STRONGLY_CONVEX else cp.Maximize(self.rho)
                self.new_objective = cp.Minimize(cp.sum([self.vars[it][-1] for it in range(self.NB_ITS)]))

            # Invalid objective
            case _:
                raise ValueError(f"Invalid objective {objective}")

    def solve(self, **kwargs):
        """
        Finalizes the constraints and solves the optimization problem.
        If new_objective is set, problem will be re-solved with warm start.

        :param solver: Solver to use. Must be one of ["MOSEK", "CLARABEL"]. Default is "MOSEK".

        :return: None
        """

        # Retrieve parameters from kwargs
        solver = kwargs.get('solver', "MOSEK")

        # Finalize the setup of the constraints
        self.__setup_constraints(**kwargs)

        # Set up and solve the problem
        problem = cp.Problem(self.objective, self.constraints)
        problem.solve(solver=solver)

        # Store outputs from the solver
        self.value = problem.value

        # If there is a new objective, re-solve
        if self.new_objective:
            # Try various values of eps until the problem is feasible
            for eps in self.EPS:
                try:
                    # Generate new constraints
                    if self.STRONGLY_CONVEX:
                        a0, e0 = self.vars[0]
                        new_constraints = [*self.constraints, a0 <= a0.value * (1 + eps)]
                    else:
                        new_constraints = [*self.constraints, self.rho >= self.rho.value - eps]

                    # Redefine the problem and solve it with warm start
                    new_problem = cp.Problem(self.new_objective, new_constraints)
                    if solver == "MOSEK":
                        new_problem.solve(solver=solver, warm_start=True, eps=1e-8 if eps == 0 else eps)
                    else:
                        E = 1e-4
                        new_problem.solve(solver=solver, warm_start=True, tol_gap_rel=E, tol_gap_abs=E, tol_feas=E)

                    self.new_value = new_problem.value
                    break
                except cp.error.SolverError:
                    # If the problem is infeasible or unsolvable, try the next value of eps
                    continue

        # Set the Lyapunov parameters
        self.__set_lyapunov_parameters()

    # From here we add some functions making it easy to analyze the results produced by the solver
    def __set_lyapunov_parameters(self, **kwargs):
        """
        Sets the Lyapunov parameters based on the values of the variables obtained from the optimization problem.
        """

        # Initialize a bag of parameters
        self.param = BagOfParameters()

        # Set all the Lyapunov parameters
        a = np.array([self.vars[k][0].value for k in range(self.NB_ITS + 1)])
        self.param.a = ParameterList(value=a, name="a_k")
        rho = self.rho.value
        self.param.d = Parameter(value=rho, name="d")
        e = np.array([self.vars[k][-1].value for k in range(self.NB_ITS)])
        self.param.e = ParameterList(value=e, name="e_k")

        # Set the bias parameter
        self.param.bias = Parameter(value=self.value, name="rate")

        # Set the variance parameter
        try:
            self.param.variance = Parameter(value=sum(e), name="variance")
            self.param.variance_sum = Parameter(value=sum(e), name="variance_sum")
            self.param.variance_avg = Parameter(value=sum(e) / self.NB_ITS, name="variance_avg")
        except Exception as e:
            self.param.variance = Parameter(value=None, name="variance")
            self.param.variance_sum = Parameter(value=None, name="variance_sum")
            self.param.variance_avg = Parameter(value=None, name="variance_avg")
