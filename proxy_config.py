"""
Утилиты для настройки HTTP/HTTPS прокси.
Читает HTTP_PROXY, HTTPS_PROXY и ALL_PROXY из переменных окружения или .env.
"""

import os
from typing import Any, Dict, Optional

import httpx
from openai import OpenAI


def get_proxy_url() -> Optional[str]:
    """Возвращает URL прокси из переменных окружения."""
    return (
        os.getenv("HTTPS_PROXY")
        or os.getenv("HTTP_PROXY")
        or os.getenv("ALL_PROXY")
    )


def get_requests_proxies() -> Optional[Dict[str, str]]:
    """Словарь прокси для библиотеки requests."""
    proxy = get_proxy_url()
    if not proxy:
        return None
    return {"http": proxy, "https": proxy}


def configure_proxy_env() -> Optional[str]:
    """
    Устанавливает стандартные переменные окружения для прокси,
    чтобы их подхватили RAGAS, langchain и другие библиотеки.
    """
    proxy = get_proxy_url()
    if not proxy:
        return None

    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.setdefault(name, proxy)
    return proxy


def create_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """Создаёт OpenAI-клиент с поддержкой прокси."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY не установлен")

    proxy_url = configure_proxy_env()
    if proxy_url:
        http_client = httpx.Client(proxy=proxy_url, timeout=60.0)
        return OpenAI(api_key=key, http_client=http_client)

    return OpenAI(api_key=key)


def get_proxy_info() -> Dict[str, Any]:
    """Информация о текущей конфигурации прокси."""
    proxy = get_proxy_url()
    return {
        "enabled": bool(proxy),
        "url": proxy,
    }
