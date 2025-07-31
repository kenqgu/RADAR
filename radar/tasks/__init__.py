import importlib
import pkgutil

# Import all modules in tasks.funcs
funcs_package = importlib.import_module("radar.tasks.funcs")
for _, module_name, is_pkg in pkgutil.iter_modules(funcs_package.__path__):
    if not is_pkg:
        importlib.import_module(f"radar.tasks.funcs.{module_name}")
