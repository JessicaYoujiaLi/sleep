import dabest
import pandas as pd


# calculating significant cells during sleep
def ci_difference_unpaired(awake_arr, nrem_arr, resamples=5000):
    """
    uses estimation stats to evaluate the difference between two sets of numbers.
    the documentation can be found here: https://acclab.github.io/DABEST-python-docs/tutorial.html

    Parameters:
    ===========
    awake_arr: numpy array
        compiled dfof of awake periods
    nrem_arr: numpy array
        compiled dfof of NREM periods
    resamples: int
        number of resamples to generate the effect size bootstrap

    Returns:
    ========
    difference: float
        this is the Cohen D effect size between the two arrays
    significance: float
        the p value for the (unpaired) student test
    """
    dabest_dict = {'awake': awake_arr, 'nrem': nrem_arr}
    dabest_df = pd.DataFrame.from_dict(dabest_dict, orient='index')
    dabest_df = dabest_df.transpose()

    # calculating the stats
    two_groups_unpaired = dabest.load(dabest_df, idx=("awake", "nrem"),
                                      resamples=resamples)
    stats = two_groups_unpaired.cohens_d
    significance = stats.results['pvalue_students_t'][0] < 0.05
    difference = stats.results['difference'][0]

    return difference, significance

def ci_difference_paired(awake_arr, nrem_arr, resamples=5000):
    """
    uses estimation stats to evaluate the difference between two sets of numbers.
    the documentation can be found here: https://acclab.github.io/DABEST-python-docs/tutorial.html

    Parameters:
    ===========
    awake_arr: numpy array
        compiled dfof of awake periods
    nrem_arr: numpy array
        compiled dfof of NREM periods
    resamples: int
        number of resamples to generate the effect size bootstrap

    Returns:
    ========
    difference: float
        this is the Cohen D effect size between the two arrays
    significance: float
        the p value for the (paired) student test
    """
    dabest_dict = {'awake': awake_arr, 'nrem': nrem_arr}
    dabest_df = pd.DataFrame.from_dict(dabest_dict, orient='index')
    dabest_df = dabest_df.transpose()

    # calculating the stats
    two_groups_paired = dabest.load(dabest_df, idx=("awake", "nrem"),
                                      resamples=resamples)
    stats = two_groups_paired.cohens_d
    significance = stats.results['pvalue_students_t'][0] < 0.05
    difference = stats.results['difference'][0]

    return difference, significance