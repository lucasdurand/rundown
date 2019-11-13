"""
Monkey patch rundown into Pandas with `pd.DataFrame.rundown=rundown.rundown_method`
Then you can make a rundown with df.rundown()
"""

from .utils import rundown as _rundown

def rundown(df, n=10):
    '''
    	What's a rundown? It's High-level summary of a pandas DataFrame, meant to render in Jupyter Notebooks.

    	"Do you have that rundown ready for me, Jim?"
    '''
    df.columns.names = [None for name in df.columns.names]
    df = df.reset_index()
    
    dfs = []
    for col in df:
        dfs+=[(col,_rundown.do_column(df,col))]

    args = [df if type(df)==_rundown.pd.DataFrame else _rundown.matplotlib_to_html(df) for (col,df) in dfs]
    
    # create the rundown, the addition data on top of the head
    rundown = _rundown.combine_to_row(*args)
    #rundown = _rundown.add_dtypes(rundown, df)
    
    # sample the dataframe and send to html
    sample = _rundown.add_dtypes(df.sample(min(n,len(df))).to_html(index=False),df)
    #sample = df.sample(min(n,len(df))).to_html(index=False)
    the_whole_thing = sample.replace('<thead>',f'<thead><tr>{rundown}</tr>')

    class Rundown:
        def _repr_html_(self):
            return the_whole_thing

    return Rundown()


_rundown.pd.DataFrame.rundown = rundown
