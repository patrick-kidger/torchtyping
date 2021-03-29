import torch
import warnings


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    torch.rand(2, names=("a",))


def pytest_configure(config):
    message = (
        "torchtyping_patch_typeguard: run with the typeguard patch. PyTest "
        "should be run twice, once with and once without this patch."
    )
    config.addinivalue_line("markers", message)
    if not config.option.torchtyping_patch_typeguard:
        if hasattr(config.option, "markexpr") and len(config.option.markexpr) > 0:
            config.option.markexpr = (
                "(" + config.option.markexpr + ") and not torchtyping_patch_typeguard"
            )
        else:
            config.option.markexpr = "not torchtyping_patch_typeguard"
