from typing import Any
from omegaconf import DictConfig, OmegaConf


def resolve_omegaconf_to_dictionary(
    conf_dict: dict[str, Any] | None | DictConfig 
) -> dict[str, Any]:
    """"""
    out_dict = {}
    if conf_dict is not None:
        out_dict = conf_dict
        if isinstance(out_dict, dict):
            out_dict = OmegaConf.create(out_dict)
        out_dict = OmegaConf.to_container(out_dict, resolve=True)
    return out_dict
