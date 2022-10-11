import matplotlib.pyplot as plt
import pandas as pd

from ..utils._param_validation import validate_params

plt.style.use("seaborn")


@validate_params({"df": [pd.core.frame.DataFrame], "tokenized": ["boolean"]})
def sequence_length_histogram(df, col, tokenized):
    """Function to plot a histogram of sequence length for a paticular column in a pandas df

    Args:
        df (pandas.core.frame.DataFrame): df in question
        col (str): name of column
        tokenized (bool): Whether or not the column text has already been tokenized
    """
    if tokenized:
        lengths = [len(t.split(" ")) for t in X["review_text"]]
    else:
        lengths = [len(s) for s in df[col]]

    plt.boxplot(lengths)

    plt.title(f"Boxplot of Sequence lengths for: {col}")
    plt.ylabel("Sequence length")
    plt.xticks([])

    plt.plot()
