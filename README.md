# rundown
A fun addition to pandas?

## Installation

`pip install rundown`

## Usage

To be run in a Jupyter Notebook, or somewhere else with nice HTML rendering of Pandas tables (i.e. not Github)

```python
import numpy as np
import pandas as pd
import rundown

sample_df = pd.DataFrame(np.random.randn(30, 4), columns=list("ABCD")) # make one!

sample_df.rundown()
```

![rundown](rundown_screenshot.png)


## Releasing

[The Release workflow](.github/workflows/release.yml) publishes to PyPI based on the version in `pyproject.toml`. It needs a `PYPI_API_TOKEN` Actions secret.

- **Merge to `main`:** if that version isn't tagged yet, it's published to PyPI and tagged (`v<version>`) with a GitHub release. To release, bump the version (`uv version --bump patch`) and merge to `main`.
- **Merge to `develop`:** a dev pre-release `<version>.dev<N>` is published (N is the workflow run number). If the version is already released, dev builds count towards the next patch version. Install one with `pip install --pre rundown`.
