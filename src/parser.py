from z3 import *
import re

def extract_variable_names(smtlib_text):
  return sorted(set(re.findall(r'\bX_\d+\b', smtlib_text)), key=lambda x: int(x.split('_')[1]))

def load_smtlib_and_parse_disjuncts(filename):
  with open(filename, 'r') as f:
    content = f.read()
  x_vars = extract_variable_names(content)
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
  opt = Optimize()
  opt.add(conjunct_formula)
  lowers = []
  uppers = []
  for name in x_var_names:
    var = var_symbols[name]
    h1 = opt.minimize(var)
    if opt.check() == sat:
      model = opt.model()
      lb = model.eval(var, model_completion=True).as_decimal(100)
      lowers.append(float(lb))
    else:
      lowers.append(None)
    opt = Optimize()
    opt.add(conjunct_formula)
    h2 = opt.maximize(var)
    if opt.check() == sat:
      model = opt.model()
      ub = model.eval(var, model_completion=True).as_decimal(100)
      uppers.append(float(ub))
    else:
      uppers.append(None)

  return lowers, uppers

def parse(path):
  disjuncts, var_symbols, var_names = load_smtlib_and_parse_disjuncts(path)
  bounds_dict = {}
  for i, disjunct in enumerate(disjuncts, start=1):
    lower, upper = get_bounds_from_conjunct(disjunct, var_symbols, var_names)
    bounds_dict[i] = (lower, upper)
  return bounds_dict
