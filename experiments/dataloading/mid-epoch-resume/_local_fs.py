"""OFF-CLUSTER TESTING ONLY -- never needed on AIR.

The SDK's ``CachingUCFuseFSLayer`` refuses any path not under ``/Volumes``. To exercise the *real*
SDK code (UCVolumeDataset, DataLoader checkpointing, UCVolumeWriter/Reader) against a local
directory, this shim disables only that prefix check. Activated by ``AIR_LAB_LOCAL_FS=1``
(see ``resumable_data.import_air_data``); everything else in the SDK runs unmodified.
"""

import importlib

_PATCHED = []
for _mod in ("databricks.air.data.fs_layer", "serverless_gpu.utils.fs_layer"):
    try:
        _m = importlib.import_module(_mod)
    except Exception:  # noqa: BLE001 - whichever package is installed
        continue
    _m.CachingUCFuseFSLayer._validate_path = lambda self, path: None
    _PATCHED.append(_mod)
