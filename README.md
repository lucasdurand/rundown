# rundown
A fun addition to pandas?

## Installation

`pip install rundown`

## Usage

To be run in a Jupyter Notebook, or somewhere else with nice HTML rendering of Pandas tables (i.e. not Github)

```python
import pandas as pd
import rundown

sample_df = pd.util.testing.makeDataFrame() # make one!

sample_df.rundown()
```

![rundown](rundown_screenshot.png)
