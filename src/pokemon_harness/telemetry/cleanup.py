"""cleanup without replacing the original failure"""

import sys
from contextlib import contextmanager


@contextmanager
def closing_safely(resource, label):
    try:
        yield resource
    finally:
        cleanup_operations([(label, resource.close)], sys.exc_info()[1])


def cleanup_operations(operations, primary_error=None):
    failures = []
    for label, operation in operations:
        try:
            operation()
        except BaseException as exc:
            failures.append((label, exc))
    for label, error in failures if primary_error is not None else failures[1:]:
        print(f"Cleanup failed ({label}): {error}", file=sys.stderr)
    if failures and primary_error is None:
        raise failures[0][1]
