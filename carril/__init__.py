"""Vision de carril con la camara del NAO (Copa NAO CCM MX 2026, Carrera Individual). Ver docs/fase_p_camara.md.

- lane_estimator_puro.EstimadorCarrilPuro: Python puro (2.7 y 3), sin dependencias. Para el NAO si no hay numpy.
- LaneFilter y LaneController tambien son Python puro.
- LaneEstimator y CameraModel usan numpy (unas 7 veces mas rapidos).
"""
from .lane_estimator_puro import EstimadorCarrilPuro  # noqa: F401
from .lane_filter import LaneFilter  # noqa: F401  (Python puro)
from .controller import LaneController  # noqa: F401  (Python puro)

try:  # en un NAO sin numpy solo queda disponible EstimadorCarrilPuro
    from .camera_model import CameraModel, transform_from_list, make_transform, FRAME_ROBOT  # noqa: F401
    from .lane_estimator import LaneEstimator, LaneEstimate  # noqa: F401
except ImportError:  # pragma: no cover
    pass
