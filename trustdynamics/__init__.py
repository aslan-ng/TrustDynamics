from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("trustdynamics")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "0.0.0"

from .model import Model
from .organization import Organization
from .technology import Technology