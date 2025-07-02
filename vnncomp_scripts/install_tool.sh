#!/bin/sh

#checking arguments
if [ "$1" != v1 ]; then
	echo "Expected first argument (version string) 'v1', got '$1'."
	exit 1
fi

echo "Installing SobolBox"

#installing tool

#Python 3.7 or higher
apt-get update && apt-get install -y software-properties-common gcc && add-apt-repository -y ppa:deadsnakes/ppa
apt-get update && apt-get install -y python3 python3-distutils python3-pip python3-apt libprotoc-dev protobuf-compiler

apt install -y psmisc #for killall, used in prepare_instance.sh

#git clone https://github.com/dassarthak18/SobolBox.git
#cd SobolBox
pip3 install -r ../requirements.txt
