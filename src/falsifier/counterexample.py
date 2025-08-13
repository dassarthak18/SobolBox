import importlib
import numpy as np
from joblib import Parallel, delayed, cpu_count
from multiprocessing import Manager, get_context
from falsifier.optimizer import sobol_samples
from falsifier.extrema_estimates import black_box
from z3 import *

def parallel_objective_eval(sess, samples, input_shape, input_name, label_name, batch_size=None):
    samples = np.asarray(samples, dtype=np.float32)
    n_samples = len(samples)
    n_jobs = cpu_count()

    if batch_size is None:
        batch_size = max(8, int(np.ceil(n_samples / n_jobs)))

    def objective(x):
        val = black_box(sess, x, input_name, label_name, input_shape)
        return val

    batches = [samples[i:i + batch_size] for i in range(0, n_samples, batch_size)]

    def evaluate_batch(batch):
        return [objective(s) for s in batch]

    results_nested = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(evaluate_batch)(batch) for batch in batches
    )

    return [val for sublist in results_nested for val in sublist]

def build_solver(n_x, n_y, smtlib_str):
    s = Solver()
    s.add(parse_smt2_string(smtlib_str))
    return s

def check_point(X_point, Y_point, n_x, n_y, smtlib_str, stop_flag):
    if stop_flag.value:
        return "unknown"
    s = build_solver(n_x, n_y, smtlib_str)
    for i, val in enumerate(X_point):
        s.add(Real(f'X_{i}') == val)
    for i, val in enumerate(Y_point):
        s.add(Real(f'Y_{i}') == val)
    if s.check() == sat:
        stop_flag.value = True
        model = s.model()
        pairs = []
        vars_in_model = [d.name() for d in model.decls()]
        x_vars = sorted([v for v in vars_in_model if v.startswith('X_')],
                        key=lambda x: int(x.split('_')[1]))
        y_vars = sorted([v for v in vars_in_model if v.startswith('Y_')],
                        key=lambda y: int(y.split('_')[1]))
        ordered_vars = x_vars + y_vars
        for var_name in ordered_vars:
            val = model[Real(var_name)]
            val = val.as_decimal(128)
            pairs.append(f"({var_name} {val})")
        return "sat\n(" + "\n ".join(pairs) + ")"
    return "unknown"

def SAT_check(X_points, Y_points, smtlib_str):
    X_points = np.asarray(X_points)
    Y_points = np.asarray(Y_points)
    n_x = X_points.shape[1]
    n_y = Y_points.shape[1]
    if "jax" in importlib.sys.modules:
        ctx = get_context("spawn")
    else:
        ctx = get_context("fork")
    with ctx.Manager() as manager:
        stop_flag = manager.Value('b', False)
        results = Parallel(n_jobs=cpu_count())(
            delayed(check_point)(X_points[i], Y_points[i], n_x, n_y, smtlib_str, stop_flag)
            for i in range(len(X_points))
        )
    sat_results = [r for r in results]
    for res in sat_results:
        if res.startswith("sat"):
            return res
    return "unknown"

def NUTS_sampler(dim, sigma, input_lb, input_ub, targets):
    import pymc as pm
    import pytensor.tensor as pt
    
    sigma2 = sigma ** 2
    n_cores = min(4, cpu_count())
    n_tune = max(100*dim, 3000)
    n_samples = max(100*dim, 2000)

    with pm.Model() as model:
        z = pm.Normal("z", mu=0, sigma=1, shape=dim)
        x = pm.Deterministic("x", input_lb + (input_ub - input_lb) * pm.math.sigmoid(z))

        def logp_fn(x_val):
            x_exp = pt.reshape(x_val, (1, -1))
            diffs = x_exp - targets
            sq_dists = pt.sum(pt.sqr(diffs), axis=1)
            logps = -0.5 * sq_dists / sigma2
            return pm.math.logsumexp(logps)

        pm.Potential("target_bias", logp_fn(x))
        trace = pm.sample(draws=n_samples, tune=n_tune, cores=n_cores, chains=n_cores, random_seed=42, target_accept=0.92, compute_convergence_checks=True, nuts_sampler="numpyro")

    NUTS_inputs = trace.posterior["x"].stack(sample=("chain", "draw")).values.T
    del model
    
    return NUTS_inputs

def CE_search(smtlib_str, sess, input_lb, input_ub, output_lb, output_ub, output_lb_inputs, output_ub_inputs, setting):
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    input_shape = sess.get_inputs()[0].shape

    optima_inputs = []
    for lb, ub in zip(output_lb_inputs, output_ub_inputs):
        optima_inputs.append(lb)
        optima_inputs.append(ub)

    optimas = parallel_objective_eval(
        sess, 
        samples=optima_inputs, 
        input_shape=input_shape, 
        input_name=input_name, 
        label_name=label_name
    )

    input_lb = np.array(input_lb)
    input_ub = np.array(input_ub)
    dim = len(input_lb)
    n_samples = min(2**15, max(4096, int(2**np.ceil(np.log2(100 * dim)))))
    unit_samples = sobol_samples(dim, n_samples)
    sobol_inputs = input_lb + unit_samples * (input_ub - input_lb)
    
    sobols = parallel_objective_eval(
        sess, 
        samples=sobol_inputs, 
        input_shape=input_shape, 
        input_name=input_name, 
        label_name=label_name
    )

    print("Checking for violations in optima.")
    result = SAT_check(optima_inputs, optimas, smtlib_str)
    if result[:3] == "sat":
        print("Safety violation found in optima.")
        return result
    
    print("Checking for violations in Sobol samples.")
    result = SAT_check(sobol_inputs, sobols, smtlib_str)
    if result[:3] == "sat":
        print("Safety violation found in Sobol samples.")
        return result
    
    if setting:
        print("Computing NUTS samples.")
        targets = np.array(optima_inputs)
        sigma = 0.1
        NUTS_inputs = NUTS_sampler(dim, sigma, input_lb, input_ub, targets)

        NUTS_outputs = parallel_objective_eval(
            sess, 
            samples=NUTS_inputs, 
            input_shape=input_shape, 
            input_name=input_name, 
            label_name=label_name
        )
        
        print("Checking for violations in NUTS samples.")
        result = SAT_check(NUTS_inputs, NUTS_outputs, smtlib_str)
        if result[:3] == "sat":
            print("Safety violation found in NUTS samples.")
            return result

    solver = build_solver(len(input_lb), len(output_lb), smtlib_str)
    for i in range(len(output_lb)):
        solver.add(Real(f'Y_{i}') >= output_lb[i])
        solver.add(Real(f'Y_{i}') <= output_ub[i])
    if str(solver.check()) == "unsat":
      print("No safety violations found.")
      return "unsat"
    
    print("Inconclusive analysis.")
    return "unknown"
