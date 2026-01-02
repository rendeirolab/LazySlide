from ._setting import settings


def default_value(name, value=None):
    if value is None:
        return getattr(settings, name)
    else:
        return value
