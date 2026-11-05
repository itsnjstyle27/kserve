# Copyright 2026 The KServe Authors.
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

import argparse

import pytest

from huggingfaceserver import Backend, parse_backend


@pytest.mark.parametrize("backend", list(Backend))
def test_parse_backend_valid(backend):
    assert parse_backend(backend.name) is backend


@pytest.mark.parametrize("value", ["vLLM", "transformers", ""])
def test_parse_backend_invalid(value):
    with pytest.raises(argparse.ArgumentTypeError, match="invalid backend"):
        parse_backend(value)


def test_backend_argument_reports_usage_error(capsys):
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="auto", type=parse_backend)

    assert parser.parse_args([]).backend is Backend.auto

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--backend", "foo"])
    assert exc_info.value.code == 2
    assert "invalid backend 'foo'" in capsys.readouterr().err
