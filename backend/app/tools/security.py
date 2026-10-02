"""Security utilities for untrusted input validation, SSRF defense, and prompt injection mitigation."""

import ipaddress
import re
import socket
from urllib.parse import urlparse
from typing import Optional, Tuple
from backend.app.config import settings


class SecurityError(Exception):
    """Raised when an operation violates security constraints."""
    pass


class URLValidator:
    """Validates URLs against SSRF, unauthorized schemes, and private network access."""

    @staticmethod
    def is_safe_url(url: str) -> Tuple[bool, Optional[str]]:
        """Validate if a URL is safe to fetch over the public internet.

        Returns (is_safe, error_reason).
        """
        try:
            parsed = urlparse(url)
        except Exception as e:
            return False, f"Malformed URL: {e}"

        # 1. Scheme check
        if not parsed.scheme or parsed.scheme.lower() not in settings.ALLOWED_URL_SCHEMES:
            return False, f"Disallowed URL scheme '{parsed.scheme}'. Only {settings.ALLOWED_URL_SCHEMES} are allowed."

        hostname = parsed.hostname
        if not hostname:
            return False, "URL lacks a valid hostname."

        # Check for obvious localhost strings
        if hostname.lower() in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
            return False, f"Access to localhost/loopback address '{hostname}' is forbidden."

        # Check for cloud metadata hostname
        if hostname.lower() in ("instance-data", "metadata.google.internal"):
            return False, f"Access to cloud metadata '{hostname}' is forbidden."

        # 2. DNS resolution and IP checks (SSRF Defense)
        try:
            # Resolve IPv4/IPv6 addresses
            addr_info = socket.getaddrinfo(hostname, None)
            for item in addr_info:
                ip_str = item[4][0]
                ip_obj = ipaddress.ip_address(ip_str)

                # Check if IP falls in private / loopback / link-local / multicast
                if (
                    ip_obj.is_private
                    or ip_obj.is_loopback
                    or ip_obj.is_link_local
                    or ip_obj.is_multicast
                    or ip_obj.is_reserved
                ):
                    return False, f"Hostname '{hostname}' resolves to restricted IP: {ip_str}"

                # Explicit check against blocked CIDRs
                for blocked_cidr in settings.BLOCKED_IP_NETWORKS:
                    if ip_obj in ipaddress.ip_network(blocked_cidr):
                        return False, f"IP {ip_str} matches blocked network {blocked_cidr}"
        except socket.gaierror:
            # Domain cannot be resolved - let caller handle network failure or block
            pass
        except Exception as e:
            return False, f"DNS resolution check failed: {e}"

        return True, None


class PromptInjectionGuard:
    """Isolates untrusted document content and detects prompt injection attempts."""

    SUSPICIOUS_PATTERNS = [
        r"(?i)(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts?|directives?)",
        r"(?i)system\s*prompt",
        r"(?i)(disregard|ignore)\s+(the\s+)?(prompt|system)",
        r"(?i)you\s+are\s+now\s+(DAN|unconstrained|jailbroken)",
        r"(?i)reveal\s+(your\s+)?(api[_\s]?key|instructions|secret)",
        r"(?i)<\|im_start\|>",
        r"(?i)<\|im_end\|>",
        r"(?i)\[INST\].*\[/INST\]",
    ]

    @classmethod
    def contains_injection_pattern(cls, text: str) -> Tuple[bool, Optional[str]]:
        """Check if untrusted text contains known adversarial prompt injection phrases."""
        for pattern in cls.SUSPICIOUS_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return True, match.group(0)
        return False, None

    @classmethod
    def sanitize_untrusted_content(cls, text: str) -> str:
        """Sanitize untrusted content by neutralizing control tokens while preserving content."""
        if not text:
            return ""
        # Neutralize common LLM prompt boundary control tokens
        text = text.replace("<|im_start|>", "[sanitized_im_start]")
        text = text.replace("<|im_end|>", "[sanitized_im_end]")
        text = text.replace("<|endoftext|>", "[sanitized_endoftext]")
        return text

    @classmethod
    def wrap_untrusted_data(cls, tag: str, content: str) -> str:
        """Wrap untrusted document text in an XML-style data block with strict isolation instructions."""
        sanitized = cls.sanitize_untrusted_content(content)
        return (
            f"<{tag} is_untrusted_external_data=\"true\">\n"
            f"{sanitized}\n"
            f"</{tag}>"
        )
