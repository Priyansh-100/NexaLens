"""
Encryption service for sensitive data at rest.

Uses Fernet (AES-128-GCM) for symmetric encryption.
Key derived from SECRET_KEY via HKDF.
"""

import base64
import os
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from nexalens.core.config import get_settings


class EncryptionService:
    """Handles encryption/decryption of sensitive configuration values."""

    def __init__(self):
        settings = get_settings()
        self._fernet = self._derive_fernet(settings.secret_key)

    def _derive_fernet(self, secret_key: str) -> Fernet:
        """Derive a Fernet key from the application secret."""
        # Use HKDF to derive a 32-byte key from the secret
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"nexalens-datasource-encryption",
            info=b"datasource-config-encryption",
        )
        key = hkdf.derive(secret_key.encode())
        fernet_key = base64.urlsafe_b64encode(key)
        return Fernet(fernet_key)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt a string and return base64-encoded ciphertext."""
        if not plaintext:
            return plaintext
        ciphertext = self._fernet.encrypt(plaintext.encode())
        return base64.urlsafe_b64encode(ciphertext).decode()

    def decrypt(self, ciphertext_b64: str) -> str:
        """Decrypt a base64-encoded ciphertext string."""
        if not ciphertext_b64:
            return ciphertext_b64
        try:
            ciphertext = base64.urlsafe_b64decode(ciphertext_b64.encode())
            plaintext = self._fernet.decrypt(ciphertext)
            return plaintext.decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {e}")

    def encrypt_dict(self, data: dict[str, Any], sensitive_keys: set[str] | None = None) -> dict[str, Any]:
        """Encrypt sensitive values in a dictionary."""
        if sensitive_keys is None:
            sensitive_keys = {
                "password", "passwd", "pwd",
                "api_key", "apikey", "access_key", "secret_key",
                "token", "auth_token", "bearer_token",
                "private_key", "cert", "certificate",
            }

        result = {}
        for key, value in data.items():
            if key.lower() in sensitive_keys and isinstance(value, str):
                result[key] = self.encrypt(value)
            elif isinstance(value, dict):
                result[key] = self.encrypt_dict(value, sensitive_keys)
            elif isinstance(value, list):
                result[key] = [
                    self.encrypt_dict(item, sensitive_keys) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def decrypt_dict(self, data: dict[str, Any], sensitive_keys: set[str] | None = None) -> dict[str, Any]:
        """Decrypt sensitive values in a dictionary."""
        if sensitive_keys is None:
            sensitive_keys = {
                "password", "passwd", "pwd",
                "api_key", "apikey", "access_key", "secret_key",
                "token", "auth_token", "bearer_token",
                "private_key", "cert", "certificate",
            }

        result = {}
        for key, value in data.items():
            if key.lower() in sensitive_keys and isinstance(value, str):
                try:
                    result[key] = self.decrypt(value)
                except ValueError:
                    # If decryption fails, assume it wasn't encrypted
                    result[key] = value
            elif isinstance(value, dict):
                result[key] = self.decrypt_dict(value, sensitive_keys)
            elif isinstance(value, list):
                result[key] = [
                    self.decrypt_dict(item, sensitive_keys) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result


encryption_service = EncryptionService()