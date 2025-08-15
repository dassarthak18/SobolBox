#runs all ACAS Xu benchmarks as a sanity check for the tool
#!/bin/sh
pip3 install -r requirements.txt

cd vnncomp_scripts
chmod u+x prepare_instance.sh
chmod u+x run_instance.sh

mkdir -p ../results
runtime_log="../runtime_log.txt"

# Clear previous log file
> "$runtime_log"
total_time=0

# Running props 1 to 4 on all models
for prop in {1..4}; do
  for i in {1..5}; do
    for j in {1..9}; do
      model="ACASXU_run2a_${i}_${j}_batch_2000.onnx"
      echo "Running model $model with property $prop"
      start_time=$(date +%s.%N)
      ./prepare_instance.sh v1 acasxu \
        ../examples/acasxu/onnx/$model \
        ../examples/acasxu/vnnlib/prop_${prop}.vnnlib
      ./run_instance.sh v1 acasxu \
        ../examples/acasxu/onnx/$model \
        ../examples/acasxu/vnnlib/prop_${prop}.vnnlib \
        ../results/result_acasxu_${i}_${j}_p${prop}.txt \
        3600
      
      end_time=$(date +%s.%N)
      elapsed=$(echo "$end_time - $start_time" | bc)
      total_time=$(echo "$total_time + $elapsed" | bc)
      
      result_file="../results/result_acasxu_${i}_${j}_p${prop}.txt"
      if [ -f "$result_file" ]; then
        status=$(head -n 1 "$result_file")
        if [ "$status" = "unknown" ]; then
          echo "x$elapsed" >> "$runtime_log"
        else
          echo "$elapsed" >> "$runtime_log"
        fi
      fi
    done
  done
done

# Running specific models with props 5 to 10
for run_info in \
  "1 1 5" \
  "1 1 6" \
  "1 9 7" \
  "2 9 8" \
  "3 3 9" \
  "4 5 10"
do
  set -- $run_info
  i=$1
  j=$2
  prop=$3
  model="ACASXU_run2a_${i}_${j}_batch_2000.onnx"

  echo "Running model $model with property $prop"
  start_time=$(date +%s.%N)
  ./prepare_instance.sh v1 acasxu \
    ../examples/acasxu/onnx/$model \
    ../examples/acasxu/vnnlib/prop_${prop}.vnnlib
  ./run_instance.sh v1 acasxu \
    ../examples/acasxu/onnx/$model \
    ../examples/acasxu/vnnlib/prop_${prop}.vnnlib \
    ../results/result_acasxu_${i}_${j}_p${prop}.txt \
    3600
  
  end_time=$(date +%s.%N)
  elapsed=$(echo "$end_time - $start_time" | bc)
  total_time=$(echo "$total_time + $elapsed" | bc)
  
  result_file="../results/result_acasxu_${i}_${j}_p${prop}.txt"
  if [ -f "$result_file" ]; then
    status=$(head -n 1 "$result_file")
    if [ "$status" = "unknown" ]; then
      echo "x$elapsed" >> "$runtime_log"
    else
      echo "$elapsed" >> "$runtime_log"
    fi
  fi
done

# Count sat/unsat/unknown
unsat=0
sat=0
unknown=0

for file in ../results/result_acasxu_*.txt; do
  if [ -f "$file" ]; then
    status=$(head -n 1 "$file")
    case "$status" in
      unsat)   ((unsat++)) ;;
      sat)     ((sat++)) ;;
      unknown) ((unknown++)) ;;
    esac
  fi
done

echo "Total instances: $((unsat + sat + unknown))"
echo "unsat: $unsat"
echo "sat: $sat"
echo "unknown: $unknown"
echo "Total Time (seconds): $total_time"

# Post-run cleanup
rm -rf ../results
rm -rf ./.sobol_cache
rm -rf ./.input_bounds
rm -rf ./output_bounds
