# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Provides lazy import of the vllm.tokenizers.mistral module.
"""
from typing import Any, TYPE_CHECKING

from vllm.utils.import_utils import LazyLoader

if TYPE_CHECKING:
    # if type checking, eagerly import the module
    import vllm.tokenizers.mistral as mt
else:
    mt = LazyLoader("mt", globals(), "vllm.tokenizers.mistral")


def is_mistral_tokenizer(obj: Any) -> bool:
    """Return true if the object is a MistralTokenizer instance."""
    cls = type(obj)
    # Check for special class attribute, this avoids importing the class to
    # do an isinstance() check.
    return bool(getattr(cls, 'IS_MISTRAL_TOKENIZER', False))
