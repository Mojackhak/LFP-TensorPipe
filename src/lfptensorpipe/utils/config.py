
from typing import Any

def get_param(cfg_dict: dict[str, dict[str, Any]], paradigm: str, section: str) -> Any:
    
    if 'default' not in cfg_dict:
        cfg_dict['default'] = {'default': None}
    cfg_subdict = cfg_dict.get(paradigm, cfg_dict['default'])
        
    if 'default' not in cfg_subdict:
        cfg_subdict['default'] = None
    out = cfg_subdict.get(section, cfg_subdict['default'])
    
    return out