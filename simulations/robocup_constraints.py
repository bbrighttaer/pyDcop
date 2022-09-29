
def coordination_constraint(*args, **kwargs):
    if 'func' in kwargs and callable(kwargs['func']):
        func = kwargs.pop('func')
        return func(*args, **kwargs)
    raise RuntimeError('func not specified')
