# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

mkdir token_completion
wget -O token_completion/train.txt https://zenodo.org/record/3628665/files/java_training_pre
wget -O token_completion/dev.txt https://zenodo.org/record/3628665/files/java_validation_pre
wget -O token_completion/test.txt https://zenodo.org/record/3628665/files/java_test_pre

