# Some commands

1. Conda Environment export
 - on Windows
    - `conda env export --no-builds | findstr -v "prefix" > conda_env_config.yml`
    - Ensure to remove certain platform specific files
        - Windows: [vc, vs2015_runtime, wincertstore]
 - On Linux
    - `conda env export --no-builds | grep -v "prefix" > conda_env_config.yml`