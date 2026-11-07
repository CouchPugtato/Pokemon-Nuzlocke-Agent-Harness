"""A lost control boundary must invalidate pending decisions."""


class ControlLost(RuntimeError):
    pass
