# do the things
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML, display
import base64
from io import BytesIO

def counts(df,col):
    vals = pd.DataFrame(df[col].fillna('NaN').value_counts() / len(df)).sort_index() * 100
    vals[col] = vals[col].round(2).astype(str) + '%'
    return vals

def do_column(df,col):
    if pd.core.dtypes.common.is_integer_dtype(df[col]) or pd.core.dtypes.common.is_string_dtype(df[col]) or pd.core.dtypes.common.is_categorical_dtype(df[col]):
        if df[col].nunique() < 10:
            total = len(df)
            return counts(df,col)
        else:
            return pd.DataFrame(df[col].describe())
    if pd.core.dtypes.common.is_numeric_dtype(df[col]):
        ax = df.hist(column=col, grid=False)[0][0]
        ax.set_title(None)
        plt.close()
        return ax
    else:
        return pd.DataFrame(df[col].describe())

def matplotlib_to_html(ax):
    fig = ax.get_figure()
    
    tmpfile = BytesIO()
    fig.set_size_inches(2,1.75)
    fig.savefig(tmpfile, format='png', facecolor=None, bbox_inches='tight')
    plt.close()
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')

    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    return html

def pandas_style(col, entry, row=''):
    index = f"<th>{row}</th>" if row else ""
    pandas_plot_html = f"""
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>{col}</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      {index}
      <td>{entry}</td>
    </tr>
  </tbody>
</table>
"""
    return pandas_plot_html

def pandas_cell_style(entry, row=''):
    #index = f"<th>{row}</th>" if row else ""
    pandas_plot_html = f"""
      <td>{entry}</td>
      """
    return pandas_plot_html


def combine_to_row(*args): #convert DataFrames to HTML and display side-by-side
    # can also display HTML Table strings
    html_str=''
    valign = "vertical-align:bottom;"
    scroll_table = 'style="display: block; max-height: 1in; overflow-y: auto"'
    for df in args:
        if type(df)==pd.DataFrame:
            #html_str += f'<td style="{valign}">{df.to_html().replace("<tbody",f"<tbody {scroll_table}")}</td>'
            html_str += f'<td style="{valign}">{df.to_html(header=False).replace("<tbody",f"<tbody {scroll_table}")}</td>'
        else:
            html_str += f'<td style="min-width:125px;{valign}">{df}</td>'
    return html_str
    #deepgreen._display_html(html_str.replace('table','table style="display:inline"'),raw=True)

def add_dtypes(html, df):
    #html = df.to_html()
    dtypes = df.dtypes
    for col in df:
        dtype = dtypes[col]
        html = html.replace(f"<th>{str(col)}</th>",f"<th>{str(col)}<br/><i>{dtype}</i></th>")
    return html
