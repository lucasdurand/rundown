from ..utils import rundown as _rundown

def rundown(df):
    '''
    	What's a rundown? It's High-level summary of a pandas DataFrame, meant to render in Jupyter Notebooks.

    	"Do you have that rundown ready for me Jim?"
    '''
    dfs = []
    for col in df:
        dfs+=[(col,_rundown.do_column(df,col))]

    args = [df if type(df)==_rundown.pd.DataFrame else _rundown.matplotlib_to_html(df[0][0]) for (col,df) in dfs]
    rundown = _rundown.combine_to_row(*args)
    rundown = _rundown.add_dtypes(rundown, df)
    sample = _rundown.add_dtypes(df.sample(10).to_html(index=False),df)
    #the_whole_thing = sample.replace('</tbody>',f'<tr>{rundown}</tr></tbody>')
    the_whole_thing = sample.replace('<thead>',f'<thead><tr>{rundown}</tr>')
    #print("WHOEL",the_whole_thing)
    _rundown.display(_rundown.HTML(the_whole_thing))