models = {}
def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    model = models[name](**kwargs)
    return model