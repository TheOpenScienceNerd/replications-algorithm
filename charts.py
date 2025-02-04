import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import t
import warnings

def confidence_interval_method(replications, alpha=0.05, desired_precision=0.05, 
                               min_rep=5, decimal_place=2):
    '''
    The confidence interval method for selecting the number of replications
    to run in a simulation.
    
    Finds the smallest number of replications where the width of the confidence
    interval is less than the desired_precision.  
    
    Returns both the number of replications and the full results dataframe.
    
    Parameters:
    ----------
    replications: arraylike
        Array (e.g. np.ndarray or list) of replications of a performance metric
        
    alpha: float, optional (default=0.05)
        procedure constructs a 100(1-alpha) confidence interval for the 
        cumulative mean.
        
    desired_precision: float, optional (default=0.05)
        Desired mean deviation from confidence interval.
        
    min_rep: int, optional (default=5)
        set to a integer > 0 and ignore all of the replications prior to it 
        when selecting the number of replications to run to achieve the desired
        precision.  Useful when the number of replications returned does not
        provide a stable precision below target.
        
    decimal_places: int, optional (default=2)
        sets the number of decimal places of the returned dataframe containing
        the results
    
    Returns:
    --------
        tuple: int, pd.DataFrame
    
    '''
    n = len(replications)
    cumulative_mean = [replications[0]]
    running_var = [0.0]
    for i in range(1, n):
        cumulative_mean.append(cumulative_mean[i-1] + \
                       (replications[i] - cumulative_mean[i-1] ) / (i+1))
        
        # running biased variance
        running_var.append(running_var[i-1] + (replications[i] 
                                               - cumulative_mean[i-1]) \
                            * (replications[i] - cumulative_mean[i]))
        
    # unbiased std dev = running_var / (n - 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        running_std = np.sqrt(running_var / np.arange(n))
    
    # half width of interval
    dof = len(replications) - 1
    t_value = t.ppf(1 - (alpha / 2),  dof)    
    with np.errstate(divide='ignore', invalid='ignore'):
        std_error = running_std / np.sqrt(np.arange(1, n+1))
        
    half_width = t_value * std_error
        
    # upper and lower confidence interval
    upper = cumulative_mean + half_width
    lower = cumulative_mean - half_width
    
    # Mean deviation
    with np.errstate(divide='ignore', invalid='ignore'):
        deviation = (half_width / cumulative_mean)
    
    # commbine results into a single dataframe
    results = pd.DataFrame([replications, cumulative_mean, 
                            running_std, lower, upper, deviation]).T
    results.columns = ['Mean', 'Cumulative Mean', 'Standard Deviation', 
                       'Lower Interval', 'Upper Interval', '% deviation']
    results.index = np.arange(1, n+1)
    results.index.name = 'replications'
    
    # get the smallest no. of reps where deviation is less than precision target
    try:
        n_reps = results.iloc[min_rep:].loc[results['% deviation'] 
                             <= desired_precision].iloc[0].name
    except:
        # no replications with desired precision
        message = 'WARNING: the replications do not reach desired precision'
        warnings.warn(message)
        n_reps = -1 

    
    return n_reps, results.round(decimal_place)
        


def plotly_confidence_interval_method(n_reps, conf_ints, metric_name, 
                                   figsize=(1200, 400)):
    """
    Interactive Plotly visualization with deviation hover information
    
    Parameters:
    ----------
    n_reps: int
        Minimum number of reps selected
    conf_ints: pandas.DataFrame
       Results from `confidence_interval_method` function
    metric_name: str
        Name of the performance measure
    figsize: tuple, optional (default=(1200,400))
        Plot dimensions in pixels (width, height)
        
    Returns:
    -------
        plotly.graph_objects.Figure
    """
    fig = go.Figure()
    
    # Calculate relative deviations [1][4]
    deviation_pct = ((conf_ints['Upper Interval'] - conf_ints['Cumulative Mean']) / 
                     conf_ints['Cumulative Mean'] * 100).round(2)

    # Confidence interval bands with hover info
    for col, color, dash in zip(['Lower Interval', 'Upper Interval'], 
                              ['lightblue', 'lightblue'],
                              ['dot', 'dot']):
        fig.add_trace(go.Scatter(
            x=conf_ints.index,
            y=conf_ints[col],
            line=dict(color=color, dash=dash),
            name=col,
            text=[f'Deviation: {d}%' for d in deviation_pct],
            hoverinfo='x+y+name+text'
        ))

    # Cumulative mean line with enhanced hover
    fig.add_trace(go.Scatter(
        x=conf_ints.index,
        y=conf_ints['Cumulative Mean'],
        line=dict(color='blue', width=2),
        name='Cumulative Mean',
        hoverinfo='x+y+name'
    ))

    # Vertical threshold line
    fig.add_shape(
        type='line',
        x0=n_reps,
        x1=n_reps,
        y0=0,
        y1=1,
        yref='paper',
        line=dict(color='red', dash='dash')
    )

    # Configure layout
    fig.update_layout(
        width=figsize[0],
        height=figsize[1],
        yaxis_title=f'Cumulative Mean: {metric_name}',
        hovermode='x unified',
        showlegend=True
    )
    
    return fig
