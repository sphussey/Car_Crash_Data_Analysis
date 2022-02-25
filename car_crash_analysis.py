
# import numpy, pandas, matplotlib and seaborn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def main():
    # print out preloaded datasets from seaborn
    # print(sns.get_dataset_names())

    # load dataset car crashes to crash_df variable
    crash_df = sns.load_dataset("car_crashes")
    #print(crash_df)

    # distribution plot - way to look at univariate distribution
    sns.displot(crash_df['not_distracted'])
    # plt.show()

    # joint plot - compare two distribution - plots scatterplot by default
    sns.jointplot(x="speeding", y='alcohol', data=crash_df, kind='scatter')
    sns.jointplot(x="speeding", y='alcohol', data=crash_df, kind='hist')
    sns.jointplot(x="speeding", y='alcohol', data=crash_df, kind='hex')
    sns.jointplot(x="speeding", y='alcohol', data=crash_df, kind='kde')
    sns.jointplot(x="speeding", y='alcohol', data=crash_df, kind='reg')
    sns.jointplot(x="speeding", y='alcohol', data=crash_df, kind='resid')

    # kde plots - distribution
    sns.kdeplot(crash_df['alcohol'])

    # pair plot - distribution plot which plots relationships across the entire dataframes numerical values
    sns.pairplot(crash_df)
    tips_df = sns.load_dataset('tips')
    sns.pairplot(tips_df, hue='sex',palette='Blues')

    # RUG PLOT = plot single column of data points in a df as sticks - see more dense num of lines where data is most common
    sns.rugplot(tips_df['tip'])

    # styling
    sns.set_style('ticks') # whitegrid, dark, white, ticks
    plt.figure(figsize=(8,4))
    sns.set_context('paper', font_scale=1.4) # talk, poster are other options


    sns.despine(left=False, bottom=False)

    sns.jointplot(x="speeding", y='alcohol', data=crash_df, kind='reg')


    # categorical plots
    sns.barplot(x='sex', y='total_bill', data=tips_df, estimator=np.cov)
    sns.countplot(x='sex',data=tips_df)

    plt.show()

if __name__ == '__main__':
    main()

