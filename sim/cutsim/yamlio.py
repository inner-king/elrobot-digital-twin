"""YAML 읽기: PyYAML(YAML 1.1)은 지수에 부호가 없는 2.5e6 같은 값을 문자열로 읽으므로 실수로 읽게 한다."""
import re
from pathlib import Path

import yaml


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?(?:[eE][-+]?[0-9]+)
               |[-+]?(?:[0-9][0-9_]*)\.[0-9_]*
               |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?
               |[-+]?\.(?:inf|Inf|INF)
               |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."),
)


def loads(text):
    return yaml.load(text, Loader=_Loader)


def load(path):
    return loads(Path(path).read_text())
