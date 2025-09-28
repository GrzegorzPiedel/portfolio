# Exchange Rates Data Quality Analysis

This project is part of my Data Analysis portfolio. It is based on a
real-world problem I solved at work: missing exchange rates in the
system caused transactions to be created with outdated values. My goal
was to identify the issue, analyze its impact, and suggest solutions.

## Project Description

The system was not loading new exchange rates since December, so all
transactions were calculated with the last available rate. This could
cause financial inconsistencies and incorrect reporting.

In this notebook, I: 
- Explored the dataset to confirm the missing
values in exchange rates
- Analyzed how often the rates were missing
- Checked which currencies and transactions were affected
- Suggested potential data quality checks to prevent similar issues
in the future

## Tools and Libraries

-   Python
-   Pandas
-   NumPy
-   Matplotlib / Seaborn (for visualization)

## How to Use

1.  Clone this repository
2.  Open the notebook `exchange rates portfolio.ipynb`
3.  Run the cells to reproduce the analysis

## Key Learnings

-   How to detect missing values in time-series data
-   How to analyze the impact of missing values on downstream processes
-   How to visualize data gaps to better communicate issues

------------------------------------------------------------------------

This project demonstrates my skills in **data analysis, problem solving,
and data quality assurance** in a real-world context.
