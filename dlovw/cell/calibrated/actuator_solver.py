"""Scaled numerical refinement for the three-dimensional actuator constraint intersection."""
import numpy as np
from scipy.optimize import minimize


def refine(desired, previous, A, lower, cap, delta):
    """Return a verified feasible projection, or None; all solver variables are normalized."""
    target, centre = desired/cap, previous/cap
    radius = delta/cap
    constraints = [dict(type='ineq', fun=lambda x:A@x-lower/cap, jac=lambda x:A),
                   dict(type='ineq', fun=lambda x:1.-x@x, jac=lambda x:-2*x),
                   dict(type='ineq', fun=lambda x:radius**2-(x-centre)@(x-centre),
                        jac=lambda x:-2*(x-centre))]
    result = minimize(lambda x:.5*((x-target)@(x-target)), centre, jac=lambda x:x-target,
                      constraints=constraints, method='SLSQP', options=dict(ftol=1e-13, maxiter=150))
    u = result.x*cap
    # Feasibility is checked independently of the solver's convergence flag.
    if (np.isfinite(u).all() and (A@u-lower).min() >= -1e-11
            and np.linalg.norm(u) <= cap+1e-11 and np.linalg.norm(u-previous) <= delta+1e-11):
        return u
    return None
