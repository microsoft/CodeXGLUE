# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

wget http://files.srl.inf.ethz.ch/data/py150_files.tar.gz

mkdir py150_files
tar -C py150_files -zxvf py150_files.tar.gz
rm py150_files.tar.gz

cd py150_files
tar -zxvf data.tar.gz
rm data.tar.gz

# python preprocess.py --base_dir=py150_files --output_dir=token_completion
