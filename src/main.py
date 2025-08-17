from memo_store import memo, clear_memo

import sys, os, pickle, hashlib
import onnxruntime as rt
from falsifier.extrema_estimates import extremum_refinement
from falsifier.counterexample import CE_search
from z3 import *

def main():
    benchmark = str(sys.argv[1])
    onnxFile = str(sys.argv[2])
    propertyFile = str(sys.argv[3])
    resultFile = str(sys.argv[4])

    with open(propertyFile) as f:
        smt = f.read()

    filename = os.path.basename(propertyFile)[:-7]
    with open(f".input_bounds/{benchmark}_{filename}.pkl", "rb") as f:
        bounds_dict = pickle.load(f)

    s = "unknown"  # default if no CE is found
    sess = rt.InferenceSession(onnxFile)
    os.makedirs(".output_bounds", exist_ok=True)
    filename = os.path.basename(onnxFile)[:-5]
    bounds_cache_file = f".output_bounds/{filename}.pkl"
    # Try to load cached output bounds
    if os.path.exists(bounds_cache_file):
        with open(bounds_cache_file, "rb") as f:
            output_bounds_cache = pickle.load(f)
    else:
        output_bounds_cache = {}

    for j in bounds_dict:
        print(f"Sub-problem {j}.")
        input_lb, input_ub = bounds_dict[j]

        if len(input_lb) > 15000:
            print("Input dimension too high, quitting gracefully.")
            with open(resultFile, 'w') as file1:
                file1.write("unknown")
            return

        print("Extracting output bounds.")
        hash_key = hashlib.md5(pickle.dumps((input_lb, input_ub))).hexdigest()
        
        if hash_key in output_bounds_cache:
            output_lb, output_ub, output_lb_inputs, output_ub_inputs = output_bounds_cache[hash_key]
            print("Output bounds extracted.")
        else:
            output_lb, output_ub, output_lb_inputs, output_ub_inputs = extremum_refinement(sess, [input_lb, input_ub])
            output_bounds_cache[hash_key] = (output_lb, output_ub, output_lb_inputs, output_ub_inputs)

        with open(resultFile, 'w') as file1:
            s = CE_search(smt, sess, input_lb, input_ub,
                          output_lb, output_ub,
                          output_lb_inputs, output_ub_inputs)
            if s[:3] == "sat":
                file1.write(s)
                with open(bounds_cache_file, "wb") as f:
                    pickle.dump(output_bounds_cache, f)
                return

    with open(bounds_cache_file, "wb") as f:
        pickle.dump(output_bounds_cache, f)
        
    # Write final result
    with open(resultFile, 'w') as file1:
        file1.write(s)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
    clear_memo()
