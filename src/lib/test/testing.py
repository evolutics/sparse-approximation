import inspect


def public_functions(module):
    return [
        member
        for member in inspect.getmembers(module, inspect.isfunction)
        if not member[0].startswith("_")
    ]
