import pytest
import torchtyping


def pytest_addoption(parser):
    parser.addoption("--patch-typeguard", action="store_true", help="Run the tests with the typeguard patcher.")
    
    
def pytest_configure(config):
    config.addinivalue_line('markers', 'patch_typeguard: run with the typeguard patch. PyTest should be run twice, once with and once without this patch.')
    if config.option.patch_typeguard:
        torchtyping.patch_typeguard()
    else:
        if hasattr(config.option, 'markexpr') and len(config.option.markexpr) > 0:
            config.option.markexpr = '(' + config.option.markexpr + ') and not patch_typeguard'
        else:
            config.option.markexpr = 'not patch_typeguard'
