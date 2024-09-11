#runs examples as a sanity check for the tool

#!/bin/sh

#sudo apt install git
#git clone https://github.com/iacs-csu-2020/INNVerS.git
#cd INNVerS

cd vnncomp_scripts
sudo chmod u+x install_tool.sh
sudo chmod u+x prepare_instance.sh
sudo chmod u+x run_instance.sh

./install_tool.sh v1

mkdir ../results

./prepare_instance.sh v1 acasxu ../examples/acasxu/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/prop_1.vnnlib
./run_instance.sh v1 acasxu ../examples/acasxu/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/prop_1.vnnlib ../results/result_acasxu_1.txt 15
./prepare_instance.sh v1 acasxu ../examples/acasxu/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/prop_2.vnnlib
./run_instance.sh v1 acasxu ../examples/acasxu/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/prop_2.vnnlib ../results/result_acasxu_2.txt 15
./prepare_instance.sh v1 acasxu ../examples/acasxu/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/prop_3.vnnlib
./run_instance.sh v1 acasxu ../examples/acasxu/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/prop_3.vnnlib ../results/result_acasxu_3.txt 15

./prepare_instance.sh v1 "test" ../examples/others/test.onnx ../examples/others/test_1.vnnlib
./run_instance.sh v1 "test" ../examples/others/test.onnx ../examples/others/test_1.vnnlib ../results/result_test_1.txt 15
./prepare_instance.sh v1 "test" ../examples/others/test.onnx ../examples/others/test_2.vnnlib
./run_instance.sh v1 "test" ../examples/others/test.onnx ../examples/others/test_2.vnnlib ../results/result_test_2.txt 15
./prepare_instance.sh v1 "test" ../examples/others/test.onnx ../examples/others/test_3.vnnlib
./run_instance.sh v1 "test" ../examples/others/test.onnx ../examples/others/test_3.vnnlib ../results/result_test_3.txt 15
./prepare_instance.sh v1 "test" ../examples/others/test.onnx ../examples/others/test_4.vnnlib
./run_instance.sh v1 "test" ../examples/others/test.onnx ../examples/others/test_4.vnnlib ../results/result_test_4.txt 15

./prepare_instance.sh v1 "test" ../examples/others/test_fixed.onnx ../examples/others/test_1.vnnlib
./run_instance.sh v1 "test" ../examples/others/test_fixed.onnx ../examples/others/test_1.vnnlib ../results/result_test_fixed_1.txt 15
./prepare_instance.sh v1 "test" ../examples/others/test_fixed.onnx ../examples/others/test_2.vnnlib
./run_instance.sh v1 "test" ../examples/others/test_fixed.onnx ../examples/others/test_2.vnnlib ../results/result_test_fixed_2.txt 15
./prepare_instance.sh v1 "test" ../examples/others/test_fixed.onnx ../examples/others/test_3.vnnlib
./run_instance.sh v1 "test" ../examples/others/test_fixed.onnx ../examples/others/test_3.vnnlib ../results/result_test_fixed_3.txt 15
./prepare_instance.sh v1 "test" ../examples/others/test_fixed.onnx ../examples/others/test_4.vnnlib
./run_instance.sh v1 "test" ../examples/others/test_fixed.onnx ../examples/others/test_4.vnnlib ../results/result_test_fixed_4.txt 15

clear

echo -n "prop_1 of acasxu: "
cat ../results/result_acasxu_1.txt
echo " "
echo -n "prop_2 of acasxu: "
cat ../results/result_acasxu_2.txt
echo " "
echo -n "prop_3 of acasxu: "
cat ../results/result_acasxu_3.txt
echo " "

echo -n "prop_1 of test: "
cat ../results/result_test_1.txt
echo " "
echo -n "prop_2 of test: "
cat ../results/result_test_2.txt
echo " "
echo -n "prop_3 of test: "
cat ../results/result_test_3.txt
echo " "
echo -n "prop_4 of test: "
cat ../results/result_test_4.txt
echo " "

echo -n "prop_1 of test_fixed: "
cat ../results/result_test_fixed_1.txt
echo " "
echo -n "prop_2 of test_fixed: "
cat ../results/result_test_fixed_2.txt
echo " "
echo -n "prop_3 of test_fixed: "
cat ../results/result_test_fixed_3.txt
echo " "
echo -n "prop_4 of test_fixed: "
cat ../results/result_test_fixed_4.txt
echo " "

rm -rf ../results
