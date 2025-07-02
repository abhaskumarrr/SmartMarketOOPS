from delta_rest_client import DeltaRestClient
from functools import lru_cache
from ..config import Settings

@lru_cache()
def get_settings():
    return Settings()

def get_delta_client():
    settings = get_settings()
    if settings.delta_exchange_api_key and settings.delta_exchange_api_secret:
        return DeltaRestClient(
            base_url=settings.delta_exchange_base_url,
            api_key=settings.delta_exchange_api_key,
            api_secret=settings.delta_exchange_api_secret
        )
    return None

