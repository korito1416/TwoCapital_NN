"""Non-invasive delta (δ) override for the delta=0.025 A/B experiment.

This module is auto-imported by CPython at interpreter startup *iff* its
directory is on PYTHONPATH (Python always tries to `import sitecustomize`).
It does NOT touch models/params.py or any production model code.  Instead it
installs a post-import hook that, once the `params` module is imported by a
model/simulation script, overwrites PARAMS["δ"] in place (the dict is shared
by reference via `from params import PARAMS`, so every consumer sees it).

Activated ONLY when the environment variable MODEL_DELTA is set to a float.
No-op otherwise, so importing this file is harmless for unrelated runs.
"""
import os
import sys

_raw = os.environ.get("MODEL_DELTA")


def _install_delta_override():
    if _raw in (None, ""):
        return
    try:
        delta_value = float(_raw)
    except (TypeError, ValueError):
        sys.stderr.write(f"[delta-override] ignoring non-numeric MODEL_DELTA={_raw!r}\n")
        return

    import importlib.abc
    import importlib.machinery

    class _ParamsPatchFinder(importlib.abc.MetaPathFinder):
        """Watches for the `params` module and patches δ right after it loads."""

        def find_spec(self, fullname, path, target=None):
            if fullname != "params":
                return None
            # Let the normal finders locate params.py, then wrap its loader.
            self_ref = self
            # Temporarily remove ourselves to avoid infinite recursion.
            try:
                sys.meta_path.remove(self_ref)
            except ValueError:
                pass
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                if self_ref not in sys.meta_path:
                    sys.meta_path.insert(0, self_ref)
            if spec is None or spec.loader is None:
                return None

            original_exec = spec.loader.exec_module

            def exec_module(module, _orig=original_exec):
                _orig(module)
                if hasattr(module, "PARAMS"):
                    old = module.PARAMS.get("δ")
                    module.PARAMS["δ"] = delta_value
                    sys.stderr.write(
                        f"[delta-override] params.PARAMS['δ']: {old} -> {delta_value} "
                        f"(MODEL_DELTA)\n"
                    )
                    sys.stderr.flush()
                # Finder stays armed for the whole process so that any re-import /
                # reload of `params` also gets the override (robust; negligible cost).

            spec.loader.exec_module = exec_module
            return spec

    import importlib.util  # noqa: E402  (kept local; only needed when active)

    sys.meta_path.insert(0, _ParamsPatchFinder())
    sys.stderr.write(f"[delta-override] armed: will set δ={delta_value} on `import params`\n")
    sys.stderr.flush()


_install_delta_override()
