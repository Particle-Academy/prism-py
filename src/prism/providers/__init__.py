"""Providers.

Only the base contract is re-exported here. Concrete providers are imported
from their own modules (``prism.providers.openai``) so that importing the
contract never drags a provider — and its transport — along with it.
"""

from __future__ import annotations

from prism.providers.base import Provider

__all__ = ["Provider"]
