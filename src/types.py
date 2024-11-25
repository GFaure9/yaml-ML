from typing import TypeAlias, Union, List, Tuple, Dict, Any
import numpy as np

ArrayLike: TypeAlias = Union[List, np.ndarray]
LoadedYAMLType: TypeAlias = Tuple[
    bool,
    str,
    Dict[str, str],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Dict[str, Any]]],
]