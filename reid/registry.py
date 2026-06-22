from .base_reid import BaseReID

REID_REGISTRY = {}


def register_reid(name):
    """Decorator to register a REID model class."""
    def decorator(cls):
        REID_REGISTRY[name] = cls
        return cls
    return decorator


def create_reid(name, **kwargs):
    """Create a REID model instance by name."""
    if name not in REID_REGISTRY:
        available = ", ".join(REID_REGISTRY.keys())
        raise ValueError(f"Unknown REID model '{name}'. Available: {available}")
    cls = REID_REGISTRY[name]
    # Pass name as model_name if not already specified
    if 'model_name' not in kwargs:
        kwargs['model_name'] = name
    return cls(**kwargs)


def list_reid_models():
    """List all registered REID model names."""
    return list(REID_REGISTRY.keys())
