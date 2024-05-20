# built-in dependencies
import re

# 3rd-party dependencies
from pydantic import BaseModel, field_validator


class IPv4(BaseModel):
    ip: str

    @field_validator('ip')
    def validate_ip(cls, v):
        pattern = re.compile(r'^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
        if not pattern.match(v):
            raise ValueError('Invalid IP address')
        return v