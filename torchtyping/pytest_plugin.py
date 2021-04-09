from .typechecker import patch_typeguard


def pytest_addoption(parser):
    unused_arg = True
    group = parser.getgroup("torchtyping")
    group.addoption(
        "--torchtyping-patch-typeguard",
        action="store_true",
        help="Run torchtyping's typeguard patch.",
    )


def pytest_configure(config):
    if config.getoption("torchtyping_patch_typeguard"):
        patch_typeguard()
