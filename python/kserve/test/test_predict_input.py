# Copyright 2021 The KServe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd

from kserve.utils.utils import get_predict_input


def test_dict_of_plain_values_is_one_row():
    result = get_predict_input({"instances": [{"a": 1, "b": 2}, {"a": 3, "b": 4}]})

    assert isinstance(result, pd.DataFrame)
    assert result[["a", "b"]].values.tolist() == [[1, 2], [3, 4]]


def test_dict_of_plain_values_uses_inputs_key_and_columns():
    result = get_predict_input(
        {"inputs": [{"b": 2.5, "a": "x"}]},
        columns=["a", "b"],
    )

    assert list(result.columns) == ["a", "b"]
    assert result.values.tolist() == [["x", 2.5]]


def test_dict_of_lists_is_still_supported():
    result = get_predict_input({"instances": [{"a": [1, 2], "b": [3, 4]}]})

    assert result[["a", "b"]].values.tolist() == [[1, 3], [2, 4]]


def test_list_of_records_is_still_supported():
    result = get_predict_input({"instances": [[{"a": 1}, {"a": 2}]]})

    assert result["a"].tolist() == [1, 2]


def test_plain_lists_are_still_an_array():
    result = get_predict_input({"instances": [[1, 2], [3, 4]]})

    assert isinstance(result, np.ndarray)
    assert result.tolist() == [[1, 2], [3, 4]]
