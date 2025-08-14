from memo_store import save_memo, clear_memo

import sys
import onnxruntime as rt
from parser import parse
from falsifier.extrema_estimates import extremum_refinement
from falsifier.counterexample import CE_search
from z3 import *

def main():
    if sys.argv[1] == "--deep":
        setting = 1
    else:
        setting = 0

    benchmark = str(sys.argv[setting+1])
    onnxFile = str(sys.argv[setting+2])
    propertyFile = str(sys.argv[setting+3])
    resultFile = str(sys.argv[setting+4])

    with open(propertyFile) as f:
        smt = f.read()

    try:
        bounds_dict = parse(propertyFile)
    except TypeError as error:
        print(str(error))
        with open(resultFile, 'w') as file1:
            file1.write("unknown")
        return

    s = "unknown"  # default if no CE is found

    for j in bounds_dict:
        print(f"Sub-problem {j}.")
        input_lb, input_ub = bounds_dict[j]

        if len(input_lb) > 15000:
            print("Input dimension too high, quitting gracefully.")
            with open(resultFile, 'w') as file1:
                file1.write("unknown")
            return

        print("Extracting output bounds.")
        sess = rt.InferenceSession(onnxFile)
        output_lb, output_ub, output_lb_inputs, output_ub_inputs = extremum_refinement(sess, [input_lb, input_ub])

        with open(resultFile, 'w') as file1:
            s = CE_search(smt, sess, input_lb, input_ub,
                          output_lb, output_ub,
                          output_lb_inputs, output_ub_inputs, setting)
            if s[:3] == "sat":
                file1.write(s)
                sys.exit(0)

    # Write final result
    with open(resultFile, 'w') as file1:
        file1.write(s)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
    save_memo()
    #clear_memo()
