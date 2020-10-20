try:
    from .utils import check_tf_version
    check_tf_version()
except:
    print(f"[WARNING]\t Augtistic could not check TensorFlow version, it should be at least 2.3!")

try:
    from .layers import *
except:
    print(f"[WARNING]\t Augtistic could not import convenience access to layers, use `import augtistic.layers` instead. Sorry.")

from .version import __version__