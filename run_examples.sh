#runs examples as a sanity check for the tool

#!/bin/sh
pip3 install -r requirements.txt

cd vnncomp_scripts
chmod u+x prepare_instance.sh
chmod u+x run_instance.sh

mkdir ../results

./prepare_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/vnnlib/prop_1.vnnlib
./run_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/vnnlib/prop_1.vnnlib ../results/result_acasxu_1.txt 600
./prepare_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/vnnlib/prop_2.vnnlib
./run_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/vnnlib/prop_2.vnnlib ../results/result_acasxu_2.txt 600
./prepare_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/vnnlib/prop_3.vnnlib
./run_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/vnnlib/prop_3.vnnlib ../results/result_acasxu_3.txt 600
./prepare_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/vnnlib/prop_4.vnnlib
./run_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/vnnlib/prop_4.vnnlib ../results/result_acasxu_4.txt 600
./prepare_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/vnnlib/prop_5.vnnlib
./run_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_1_1_batch_2000.onnx ../examples/acasxu/vnnlib/prop_5.vnnlib ../results/result_acasxu_5.txt 600

./prepare_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_5_9_batch_2000.onnx ../examples/acasxu/vnnlib/prop_6.vnnlib
./run_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_5_9_batch_2000.onnx ../examples/acasxu/vnnlib/prop_6.vnnlib ../results/result_acasxu_6.txt 600
./prepare_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_5_9_batch_2000.onnx ../examples/acasxu/vnnlib/prop_7.vnnlib
./run_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_5_9_batch_2000.onnx ../examples/acasxu/vnnlib/prop_7.vnnlib ../results/result_acasxu_7.txt 600
./prepare_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_5_9_batch_2000.onnx ../examples/acasxu/vnnlib/prop_8.vnnlib
./run_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_5_9_batch_2000.onnx ../examples/acasxu/vnnlib/prop_8.vnnlib ../results/result_acasxu_8.txt 600
./prepare_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_5_9_batch_2000.onnx ../examples/acasxu/vnnlib/prop_9.vnnlib
./run_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_5_9_batch_2000.onnx ../examples/acasxu/vnnlib/prop_9.vnnlib ../results/result_acasxu_9.txt 600
./prepare_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_5_9_batch_2000.onnx ../examples/acasxu/vnnlib/prop_10.vnnlib
./run_instance.sh v1 acasxu ../examples/acasxu/onnx/ACASXU_run2a_5_9_batch_2000.onnx ../examples/acasxu/vnnlib/prop_10.vnnlib ../results/result_acasxu_10.txt 600

./prepare_instance.sh v1 traffic ../examples/traffic_signs_recognition/onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx ../examples/traffic_signs_recognition/vnnlib/model_30_idx_11985_eps_1.00000.vnnlib
./run_instance.sh v1 traffic ../examples/traffic_signs_recognition/onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx ../examples/traffic_signs_recognition/vnnlib/model_30_idx_11985_eps_1.00000.vnnlib ../results/result_traffic_1.txt 600
./prepare_instance.sh v1 traffic ../examples/traffic_signs_recognition/onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx ../examples/traffic_signs_recognition/vnnlib/model_30_idx_11985_eps_10.00000.vnnlib
./run_instance.sh v1 traffic ../examples/traffic_signs_recognition/onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx ../examples/traffic_signs_recognition/vnnlib/model_30_idx_11985_eps_10.00000.vnnlib ../results/result_traffic_2.txt 600
./prepare_instance.sh v1 traffic ../examples/traffic_signs_recognition/onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx ../examples/traffic_signs_recognition/vnnlib/model_30_idx_11985_eps_15.00000.vnnlib
./run_instance.sh v1 traffic ../examples/traffic_signs_recognition/onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx ../examples/traffic_signs_recognition/vnnlib/model_30_idx_11985_eps_15.00000.vnnlib ../results/result_traffic_3.txt 600
./prepare_instance.sh v1 traffic ../examples/traffic_signs_recognition/onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx ../examples/traffic_signs_recognition/vnnlib/model_30_idx_11985_eps_3.00000.vnnlib
./run_instance.sh v1 traffic ../examples/traffic_signs_recognition/onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx ../examples/traffic_signs_recognition/vnnlib/model_30_idx_11985_eps_3.00000.vnnlib ../results/result_traffic_4.txt 600
./prepare_instance.sh v1 traffic ../examples/traffic_signs_recognition/onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx ../examples/traffic_signs_recognition/vnnlib/model_30_idx_11985_eps_5.00000.vnnlib
./run_instance.sh v1 traffic ../examples/traffic_signs_recognition/onnx/3_30_30_QConv_16_3_QConv_32_2_Dense_43_ep_30.onnx ../examples/traffic_signs_recognition/vnnlib/model_30_idx_11985_eps_5.00000.vnnlib ../results/result_traffic_5.txt 600

clear

echo -n "prop_1 of acasxu run2a_1_1_batch_2000: "
cat ../results/result_acasxu_1.txt
echo " "
echo -n "prop_2 of acasxu run2a_1_1_batch_2000: "
cat ../results/result_acasxu_2.txt
echo " "
echo -n "prop_3 of acasxu run2a_1_1_batch_2000: "
cat ../results/result_acasxu_3.txt
echo " "
echo -n "prop_4 of acasxu run2a_1_1_batch_2000: "
cat ../results/result_acasxu_4.txt
echo " "
echo -n "prop_5 of acasxu run2a_1_1_batch_2000: "
cat ../results/result_acasxu_5.txt
echo " "

echo -n "prop_6 of acasxu run2a_5_9_batch_2000: "
cat ../results/result_acasxu_6.txt
echo " "
echo -n "prop_7 of acasxu run2a_5_9_batch_2000: "
cat ../results/result_acasxu_7.txt
echo " "
echo -n "prop_8 of acasxu run2a_5_9_batch_2000: "
cat ../results/result_acasxu_8.txt
echo " "
echo -n "prop_9 of acasxu run2a_5_9_batch_2000: "
cat ../results/result_acasxu_9.txt
echo " "
echo -n "prop_10 of acasxu run2a_5_9_batch_2000: "
cat ../results/result_acasxu_10.txt
echo " "

echo -n "prop_1 of traffic: "
cat ../results/result_traffic_1.txt
echo " "
echo -n "prop_2 of traffic: "
cat ../results/result_traffic_2.txt
echo " "
echo -n "prop_3 of traffic: "
cat ../results/result_traffic_3.txt
echo " "
echo -n "prop_4 of traffic: "
cat ../results/result_traffic_4.txt
echo " "
echo -n "prop_5 of traffic: "
cat ../results/result_traffic_5.txt
echo " "

rm -rf ../results
