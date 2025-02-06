import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.stats import t
import warnings

class OnlineStatistics:

    def __init__(self, data=None, alpha=0.1, decimal_places=2):
        """
        Welford’s algorithm for computing a running sample mean and
        variance. Allowing computation of CIs.

        This is a robust, accurate and old(ish) approach (1960s) that
        I first read about in Donald Knuth’s art of computer programming vol 2.

        Params:
        -------
        data: array-like, optional (default = None)
            Contains an initial data sample.

        alpha: float
            To compute 100(1 - alpha) confidence interval

        decimal_places: int, optional (default=2)
            Summary decimal places.
        """

        self.n = 0
        self.mean = None
        # sum of squares of differences from the current mean
        self._sq = None
        self.alpha = alpha

        if isinstance(data, np.ndarray):
            for x in data:
                self.update(x)

        self.dp = decimal_places

    @property
    def variance(self):
        """
        Sample variance of data
        Sum of squares of differences from the current mean divided by n - 1
        """
        return self._sq / (self.n - 1)

    @property
    def std(self):
        """
        Standard deviation of data
        """
        return np.sqrt(self.variance)

    @property
    def std_error(self):
        """
        Standard error of the mean
        """
        return self.std / np.sqrt(self.n)

    @property
    def half_width(self):
        """
        Confidence interval half width
        """
        dof = self.n - 1
        t_value = t.ppf(1 - (self.alpha / 2), dof)
        return t_value * self.std_error

    @property
    def lci(self):
        """
        Lower confidence interval bound
        """
        return self.mean - self.half_width

    @property
    def uci(self):
        """
        Lower confidence interval bound
        """
        return self.mean + self.half_width

    @property
    def deviation(self):
        """
        Precision of the confidence interval expressed as the
        percentage deviation of the half width from the mean.
        """
        return self.half_width / self.mean

    def update(self, x):
        """
        Running update of mean and variance implemented using Welford's
        algorithm (1962).

        See Knuth. D `The Art of Computer Programming` Vol 2. 2nd ed. Page 216.

        Params:
        ------
        x: float
            A new observation
        """
        self.n += 1

        # init values
        if self.n == 1:
            self.mean = x
            self._sq = 0
        else:
            # compute the updated mean
            updated_mean = self.mean + ((x - self.mean) / self.n)

            # update the sum of squares of differences from the current mean
            self._sq += (x - self.mean) * (x - updated_mean)

            # update the tracked mean
            self.mean = updated_mean


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
    
    stdev = [np.nan] * 2 
    lower = [np.nan] * 2
    upper = [np.nan] * 2
    dev = [np.nan] * 2

   
    # online statistical and log first cumulative mean
    stats = OnlineStatistics(alpha=alpha, data=replications[:2])
    cumulative_mean = [replications[0], stats.mean]
    
    for i in range(2, n):
        stats.update(replications[i])
        cumulative_mean.append(stats.mean)
        stdev.append(stats.std)
        lower.append(stats.lci)
        upper.append(stats.uci)
        dev.append(stats.deviation)
                    
    # combine results into a single dataframe
    results = pd.DataFrame([replications, cumulative_mean, 
                            stdev, lower, upper, dev]).T
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
