



$ PYTHONPATH=/path/to/relax python prepare_tuning.py

# Tune with TVM-main. Static part
$ PYTHONPATH=/path/to/tvm python dolly_tune_static.py
$ PYTHONPATH=/path/to/tvm python dolly_make_dispatch.py

# Tune with RELAX/UNITY. Dynamic part
$ PYTHONPATH=/path/to/relax python dolly_tune_dynamic.py