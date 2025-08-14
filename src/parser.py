from z3 import *
import re, pickle, os

def extract_variable_names(smtlib_text):
    return sorted(set(re.findall(r'\bX_\d+\b', smtlib_text)), key=lambda x: int(x.split('_')[1]))

def load_smtlib_and_parse_disjuncts(filename):
    with open(filename, 'r') as f:
        content = f.read()
    x_vars = extract_variable_names(content)
    count_x_vars = sum(1 for x in x_vars if x.startswith("X_"))
    if count_x_vars > 9250:
        raise TypeError("Input dimension too high, skipping benchmark.")
    input_vars = []
    var_symbols = {name: Real(name) for name in x_vars}
    full_solver = Solver()
    full_solver.from_string(content)
    disjuncts = []
    for a in full_solver.assertions():
        if is_or(a) and all('X_' in str(c) for c in a.children()[0].children()):
            disjuncts = a.children()
            break
    if not disjuncts:
        disjuncts = [And(full_solver.assertions())]

    return disjuncts, var_symbols, x_vars

def get_bounds_from_conjunct(conjunct_formula, var_symbols, x_var_names):
    lowers = []
    uppers = []

    opt_min = Optimize()
    opt_min.add(conjunct_formula)
    for name in x_var_names:
        opt_min.minimize(var_symbols[name])

    opt_max = Optimize()
    opt_max.add(conjunct_formula)
    for name in x_var_names:
        opt_max.maximize(var_symbols[name])

    if opt_min.check() == sat:
        min_model = opt_min.model()
        for name in x_var_names:
            val = min_model.eval(var_symbols[name], model_completion=True)
            lowers.append(float(val.as_decimal(100)))
    else:
        lowers = [None] * len(x_var_names)

    if opt_max.check() == sat:
        max_model = opt_max.model()
        for name in x_var_names:
            val = max_model.eval(var_symbols[name], model_completion=True)
            uppers.append(float(val.as_decimal(100)))
    else:
        uppers = [None] * len(x_var_names)

    return lowers, uppers

def parse(path):
    print("Extracting input bounds.")
    disjuncts, var_symbols, var_names = load_smtlib_and_parse_disjuncts(path)
    bounds_dict = {}
    for i, disjunct in enumerate(disjuncts, start=1):
        lower, upper = get_bounds_from_conjunct(disjunct, var_symbols, var_names)
        bounds_dict[i] = (lower, upper)
    print("Input bounds extracted.")
    return bounds_dict

benchmark    = str(sys.argv[1])
propertyFile = str(sys.argv[2])

with open(propertyFile) as f:
    smt = f.read()

try:
    os.makedirs(".input_bounds", exist_ok=True)
    bounds_dict = parse(propertyFile)
    filename = os.path.basename(propertyFile)[:-7]
    with open(f".input_bounds/{benchmark}_{filename}.pkl", "wb") as f:
        pickle.dump(bounds_dict, f)
except TypeError as error:
    print(str(error))
    sys.exit(42) # Becausee 42 is the answer to the universe
