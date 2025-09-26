# Oscars Data Analysis

## Project overview
This project focuses on extracting, cleaning, analyzing, and visualizing data from historical Oscar ceremonies. The dataset is parsed from HTML pages (archived Oscars data), transformed into structured formats (CSV, JSON, SQLite), and enriched with exploratory analysis and visualizations.

## Key features
- **Web scraping (local HTML parsing)** using BeautifulSoup
- **Data cleaning and transformation**: standardizing person/movie names, handling missing data
- **Multiple output formats**:
  - CSV (`oscars_GP.csv`)
  - JSON (`oscars_GP-structured.json`)
  - Relational SQLite database (`oscars_relational.sqlite`)
- **Exploratory Data Analysis (EDA)**:
  - Most nominated people, movies, and countries
  - Nominations across decades
  - Categories with the most nominees
- **Visualizations** with Matplotlib:
  - Pie chart of most nominated individuals
  - Bar charts for movies, countries, and categories
  - Timeline of nominations per decade

## Tools & libraries
- **Python**
- **BeautifulSoup** for parsing HTML
- **Pandas** for data manipulation
- **Matplotlib** for visualization
- **SQLite3** for relational database schema

## Example insights
- Identification of the most frequently nominated individuals
- Analysis of international recognition in the "Foreign Language Film" category
- Historical trends of nominations per decade
- Most awarded actors/actresses in Oscars history

## Potential extensions
- Build an API or web app for interactive Oscars exploration
- Create dashboards for dynamic visual analysis
- Develop ML models to predict future winners
- Current version does not implement advanced error handling or retry mechanisms. These could be added in the future for better robustness, especially in case of API or network failures

## How to run
1. Clone this repository
2. Place the `oscars.zip` archive in the working directory
3. Run the main notebook/script (`oscars_analysis.ipynb` or `.py`)
4. Explore the generated outputs: CSV, JSON, SQLite, and plots
