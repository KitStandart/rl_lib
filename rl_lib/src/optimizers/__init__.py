try:
    import tensorflow_addons as tfa
except ImportError:
    try:
        import subprocess
        import sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "tensorflow_addons"])
        import tensorflow_addons
    except ImportError:
        print("Не удалось установить и импортировать TENSORFLOW_ADDONS")
        raise SystemExit(1)
