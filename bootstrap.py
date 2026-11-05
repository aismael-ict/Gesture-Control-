"""
Bootstrap environment loader for HandControl Next-Gen.
Automatically sets up sys.path and PyInstaller PYZ archive loader if running in standard Python.
"""
import os
import sys
import marshal
import types

_initialized = False

def init_environment():
    global _initialized
    if _initialized:
        return
    _initialized = True

    base_dir = os.path.dirname(os.path.abspath(__file__))
    internal_dir = os.path.join(base_dir, '_internal')
    pyz_path = os.path.join(base_dir, 'extracted_PYZ.pyz')

    # If running inside PyInstaller frozen app, standard environment is already loaded
    if getattr(sys, 'frozen', False):
        return

    # Add _internal directory to sys.path
    if os.path.isdir(internal_dir) and internal_dir not in sys.path:
        sys.path.insert(0, internal_dir)
        sys._MEIPASS = internal_dir

    # If extracted_PYZ.pyz exists, mount PyInstaller importers
    if os.path.isfile(pyz_path) and not hasattr(sys, '_pyinstaller_pyz'):
        try:
            from PyInstaller.archive.readers import CArchiveReader
            exe_path = os.path.join(base_dir, 'HandControl.exe')
            if os.path.isfile(exe_path):
                arch = CArchiveReader(exe_path)
                sys._pyinstaller_pyz = pyz_path

                def load_pyi_mod(name):
                    mod_data = arch.extract(name)
                    code = marshal.loads(mod_data)
                    mod = types.ModuleType(name)
                    exec(code, mod.__dict__)
                    sys.modules[name] = mod
                    return mod

                load_pyi_mod('pyimod01_archive')
                pyimod02 = load_pyi_mod('pyimod02_importers')
                pyimod02.install()
        except Exception as e:
            # If PyInstaller archive loader cannot be loaded, rely on standard imports
            pass

init_environment()

