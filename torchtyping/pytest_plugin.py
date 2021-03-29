from .typechecker import patch_typeguard


def pytest_addoption(parser):
    group = parser.getgroup("torchtyping")
    group.addoption("--enable-torchtyping", action="store_true", help="Enable torchtyping's typeguard integration.")
    
    
def pytest_configure(config):
    if config.getoption("enable_torchtyping"):
        patch_typeguard()
