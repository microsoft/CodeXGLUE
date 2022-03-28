# coding=utf-8
# Copyright 2022 Kevin Jesse.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Introduction to the typing task"""

import json

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """
"""

_DESCRIPTION = """\
A type inference dataset. 
"""

_URL = ""
_TRAINING_FILE0 = "../dataset/train0.jsonl"
_TRAINING_FILE1 = "../dataset/train1.jsonl"
_TRAINING_FILE2 = "../dataset/train2.jsonl"
_DEV_FILE = "../dataset/valid.jsonl"
_TEST_FILE = "../dataset/test.jsonl"
_VOCAB_FILE = "../dataset/vocab_50000.txt"  # your models vocabulary
_HOMEPAGE = "https://huggingface.co/datasets/kevinjesse/ManyTypes4TypeScript"
_VERSION = datasets.Version("1.0.0")


class ManyTypes4TypeScript2022Config(datasets.BuilderConfig):
    """BuilderConfig for ManyTypes4TypeScript 2022"""

    def __init__(self, **kwargs):
        """BuilderConfig for ManyTypes4TypeScript 2022.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(ManyTypes4TypeScript2022Config, self).__init__(**kwargs)


class ManyTypes4TypeScript(datasets.GeneratorBasedBuilder):
    """ManyTypes4TypeScript 2022 dataset."""

    BUILDER_CONFIGS = [
        ManyTypes4TypeScript2022Config(name="ManyTypes4TypeScript", version=_VERSION,
                                       description="ManyTypes4TypeScript dataset"),
    ]

    def _load_vocab(self):
        vocab_manager = datasets.utils.download_manager.DownloadManager()
        download_vocab = vocab_manager.download_and_extract({"vocab": f"{_URL}{_VOCAB_FILE}"})
        with open(download_vocab['vocab'], 'r') as file:
            vocab = file.readlines()
            vocab = [line.rstrip() for line in vocab]
            return vocab

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "labels": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=self._load_vocab()
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train0": f"{_URL}{_TRAINING_FILE0}",
            "train1": f"{_URL}{_TRAINING_FILE1}",
            "train2": f"{_URL}{_TRAINING_FILE2}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": [downloaded_files["train0"],
                                                                                         downloaded_files["train1"],
                                                                                         downloaded_files["train2"]]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                    gen_kwargs={"filepaths": [downloaded_files["dev"], ]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepaths": [downloaded_files["test"], ]}),
        ]

    def load_jsonl(self, input_path):
        """
        Read list of objects from a JSON lines file.
        """
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.rstrip('\n|\r')))
        # print('Loaded {} records from {}'.format(len(data), input_path))
        return data

    def _generate_examples(self, filepaths):
        """
        Generates examples from train, test, valid, files.
        """
        guid = 0
        for filepath in filepaths:
            logger.info("⏳ Generating examples from = %s", filepath)
            for line in self.load_jsonl(filepath):
                yield guid, {
                    "id": str(guid),
                    "tokens": line['tokens'],
                    "labels": line['labels'],
                }
                guid += 1
