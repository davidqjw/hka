import re
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import yaml
import random
import datetime

class Config:
    def __init__(self, config_file_path=None, config_dict={}):
        self.yaml_loader = self._build_yaml_loader()
        self.file_config = self._load_file_config(config_file_path)
        self.variable_config = config_dict
        self.final_config = self._merge_external_config()
        self._set_additional_key()
        self._init_device()

    def _build_yaml_loader(self):
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        return loader
    
    def _load_file_config(self, config_file_path: str):
        file_config = dict()
        if config_file_path:
            with open(config_file_path, "r", encoding="utf-8") as f:
                file_config.update(yaml.load(f.read(), Loader=self.yaml_loader))
        return file_config
    
    @staticmethod
    def _update_dict(old_dict: dict, new_dict: dict):
        # Update the original update method of the dictionary:
        # If there is the same key in `old_dict` and `new_dict`, and value is of type dict, update the key in dict

        same_keys = []
        for key, value in new_dict.items():
            if key in old_dict and isinstance(value, dict):
                same_keys.append(key)
        for key in same_keys:
            old_item = old_dict[key]
            new_item = new_dict[key]
            old_item.update(new_item)
            new_dict[key] = old_item

        old_dict.update(new_dict)
        return old_dict

    def _merge_external_config(self):
        external_config = dict()
        external_config = self._update_dict(external_config, self.file_config)
        external_config = self._update_dict(external_config, self.variable_config)

        return external_config

    def _set_additional_key(self):
        pass

    def _init_device(self):
        gpu_id = self.final_config["gpu_id"]
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        try:
            # import pynvml 
            # pynvml.nvmlInit()
            # gpu_num = pynvml.nvmlDeviceGetCount()
            import torch
            gpu_num = torch.cuda.device_count()
        except:
            gpu_num = 0
        self.final_config['gpu_num'] = gpu_num
        if gpu_num > 0:
            self.final_config["device"] = "cuda"
        else:
            self.final_config['device'] = 'cpu'

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config[key] = value
    
    def __getitem__(self, item):
        return self.final_config.get(item)
    
    def __contains__(self, key):
        return key in self.final_config
