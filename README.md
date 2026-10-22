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

Bump `version` in `pyproject.toml` (`uv version --bump patch`), then publish a GitHub release tagged with that version (e.g. `v0.1.5`). [The Publish workflow](.github/workflows/publish.yml) checks that the tag matches, builds the package and uploads it to PyPI. It needs a `PYPI_API_TOKEN` Actions secret.
