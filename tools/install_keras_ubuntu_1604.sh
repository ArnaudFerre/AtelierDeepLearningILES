#!/bin/bash

sudo apt-get install python3-setuptools python3-numpy python3-scipy python3-dev

cd /tmp/

wget "https://github.com/fchollet/keras/tarball/2.0.6" -O - 2> /dev/null | tar xvz

cd fchollet-keras-f120a56/

sudo python3 ./setup.py install

mkdir -p ~/.keras
cat > ~/.keras/keras.json << EOF
{
    "image_data_format": "channels_last",
    "epsilon": 1e-07,
    "floatx": "float32",
    "backend": "theano"
}
EOF
