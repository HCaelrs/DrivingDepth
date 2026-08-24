# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

WHITE = "\033[97m"
RESET = "\033[0m"

LOG_LEVELS = {"ERROR": 0, "WARN": 1, "INFO": 2, "DEBUG": 3}


class Logger:
    """INFO-only logger; set DA3_LOG_LEVEL=ERROR/WARN to silence it."""

    def __init__(self):
        level = os.environ.get("DA3_LOG_LEVEL", "INFO").upper()
        self.level = LOG_LEVELS.get(level, LOG_LEVELS["INFO"])

    def info(self, *args, **kwargs):
        if self.level >= LOG_LEVELS["INFO"]:
            msg = " ".join(str(arg) for arg in args)
            print(f"{WHITE}[INFO ] {msg}{RESET}", **kwargs)


logger = Logger()

__all__ = ["logger"]
