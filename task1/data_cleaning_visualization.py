import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class MovieDataAnalyzer:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.df_cleaned = None

    def load_data(self):
        print("="*70)
        print("LOADING DATA")
        print("="*70)
        self.df = pd.read_csv(self.filepath)
        print(f"✓ Dataset loaded: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        return self.df

    def explore_data(self):
        print("\n" + "="*70)
        print("DATA EXPLORATION")
        print("="*70)

        print("\nFirst 5 rows:")
        print(self.df.head())

        print("\nDataset Info:")
        print(self.df.info())

        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct.round(2)})
        print(missing_df[missing_df['Count'] > 0])

    def clean_data(self):
        print("\n" + "="*70)
        print("DATA CLEANING")
        print("="*70)

        self.df_cleaned = self.df.copy()
        missing_before = self.df_cleaned.isnull().sum().sum()

        if self.df_cleaned['overview'].isnull().any():
            self.df_cleaned['overview'].fillna('No description available', inplace=True)
            print("✓ Filled missing overviews")

        if self.df_cleaned['release_date'].isnull().any():
            before = len(self.df_cleaned)
            self.df_cleaned.dropna(subset=['release_date'], inplace=True)
            print(f"✓ Removed {before - len(self.df_cleaned)} rows with missing dates")

        numeric_cols = ['popularity', 'vote_average', 'vote_count']
        for col in numeric_cols:
            if self.df_cleaned[col].isnull().any():
                self.df_cleaned[col].fillna(self.df_cleaned[col].median(), inplace=True)

        print(f"Missing values: {missing_before} → {self.df_cleaned.isnull().sum().sum()}")

        print("\nHandling Duplicates...")
        duplicates = self.df_cleaned.duplicated().sum()
        if duplicates > 0:
            self.df_cleaned.drop_duplicates(inplace=True)
            print(f"✓ Removed {duplicates} duplicates")

        print("\nHandling Outliers...")
        for col in ['popularity', 'vote_count']:
            Q1 = self.df_cleaned[col].quantile(0.25)
            Q3 = self.df_cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((self.df_cleaned[col] < lower_bound) |
                       (self.df_cleaned[col] > upper_bound)).sum()
            self.df_cleaned[col] = self.df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)
            print(f"✓ Capped {outliers} outliers in {col}")

        print("\nConverting Data Types...")
        self.df_cleaned['release_date'] = pd.to_datetime(self.df_cleaned['release_date'], errors='coerce')
        self.df_cleaned['year'] = self.df_cleaned['release_date'].dt.year
        self.df_cleaned['month'] = self.df_cleaned['release_date'].dt.month

        print("\nCreating Additional Features...")
        self.df_cleaned['rating_category'] = pd.cut(
            self.df_cleaned['vote_average'],
            bins=[0, 5, 7, 8, 10],
            labels=['Poor', 'Average', 'Good', 'Excellent']
        )

        print(f"\n✅ Cleaning completed: {self.df_cleaned.shape[0]} rows × {self.df_cleaned.shape[1]} columns")
        return self.df_cleaned

    def create_visualizations(self):
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)

        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')

        print("\nCreating visualization 1/8...")
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(self.df_cleaned['vote_average'], bins=30, kde=True, color='skyblue')
        plt.title('Distribution of Movie Ratings', fontsize=14, fontweight='bold')
        plt.xlabel('Vote Average')
        plt.ylabel('Frequency')
        plt.subplot(1, 2, 2)
        sns.boxplot(y=self.df_cleaned['vote_average'], color='lightcoral')
        plt.title('Boxplot of Movie Ratings', fontsize=14, fontweight='bold')
        plt.ylabel('Vote Average')
        plt.tight_layout()
        plt.savefig('visualizations/1_vote_average_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Creating visualization 2/8...")
        plt.figure(figsize=(10, 6))
        rating_counts = self.df_cleaned['rating_category'].value_counts()
        colors = ['#ff6b6b', '#feca57', '#48dbfb', '#1dd1a1']
        plt.pie(rating_counts, labels=rating_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
        plt.title('Distribution of Movies by Rating Category', fontsize=14, fontweight='bold')
        plt.savefig('visualizations/2_rating_categories.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Creating visualization 3/8...")
        plt.figure(figsize=(14, 6))
        movies_per_year = self.df_cleaned['year'].value_counts().sort_index()
        plt.bar(movies_per_year.index, movies_per_year.values, color='steelblue', alpha=0.7)
        plt.title('Number of Movies Released Per Year', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Number of Movies')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.savefig('visualizations/3_movies_per_year.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Creating visualization 4/8...")
        plt.figure(figsize=(12, 6))
        plt.scatter(self.df_cleaned['popularity'], self.df_cleaned['vote_average'],
                   alpha=0.5, c=self.df_cleaned['vote_count'], cmap='viridis', s=50)
        plt.colorbar(label='Vote Count')
        plt.title('Movie Popularity vs Rating', fontsize=14, fontweight='bold')
        plt.xlabel('Popularity')
        plt.ylabel('Vote Average')
        plt.grid(alpha=0.3)
        plt.savefig('visualizations/4_popularity_vs_rating.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Creating visualization 5/8...")
        top_rated = self.df_cleaned.nlargest(20, 'vote_average')[['title', 'vote_average']]
        plt.figure(figsize=(12, 8))
        sns.barplot(data=top_rated, y='title', x='vote_average', palette='rocket')
        plt.title('Top 20 Highest Rated Movies', fontsize=14, fontweight='bold')
        plt.xlabel('Vote Average')
        plt.ylabel('Movie Title')
        plt.tight_layout()
        plt.savefig('visualizations/5_top_rated_movies.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Creating visualization 6/8...")
        plt.figure(figsize=(12, 6))
        monthly_releases = self.df_cleaned['month'].value_counts().sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        plt.plot(monthly_releases.index, monthly_releases.values,
                marker='o', linewidth=2, markersize=8, color='crimson')
        plt.title('Movies Released by Month', fontsize=14, fontweight='bold')
        plt.xlabel('Month')
        plt.ylabel('Number of Movies')
        plt.xticks(range(1, 13), month_names)
        plt.grid(alpha=0.3)
        plt.savefig('visualizations/6_monthly_releases.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Creating visualization 7/8...")
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.histplot(self.df_cleaned['vote_count'], bins=50, kde=True, color='mediumseagreen')
        plt.title('Distribution of Vote Counts', fontsize=14, fontweight='bold')
        plt.xlabel('Vote Count')
        plt.ylabel('Frequency')
        plt.subplot(1, 2, 2)
        sns.violinplot(y=self.df_cleaned['vote_count'], color='plum')
        plt.title('Vote Count Distribution (Violin)', fontsize=14, fontweight='bold')
        plt.ylabel('Vote Count')
        plt.tight_layout()
        plt.savefig('visualizations/7_vote_count_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Creating visualization 8/8...")
        plt.figure(figsize=(10, 8))
        numeric_cols = ['popularity', 'vote_average', 'vote_count', 'year']
        correlation = self.df_cleaned[numeric_cols].corr()
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=1, fmt='.2f')
        plt.title('Correlation Matrix of Numeric Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('visualizations/8_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("\n✅ All visualizations created!")
        print("   Saved in: visualizations/")

    def generate_report(self):
        print("\n" + "="*70)
        print("GENERATING ANALYSIS REPORT")
        print("="*70)

        report = []
        report.append("="*70)
        report.append("MOVIE DATASET ANALYSIS REPORT")
        report.append("="*70)
        report.append(f"\nGenerated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nDataset: movies.csv")
        report.append(f"Total Movies Analyzed: {len(self.df_cleaned)}")

        report.append("\n" + "-"*70)
        report.append("KEY FINDINGS")
        report.append("-"*70)

        report.append("\n1. RATING STATISTICS")
        report.append(f"   • Average Rating: {self.df_cleaned['vote_average'].mean():.2f}/10")
        report.append(f"   • Median Rating: {self.df_cleaned['vote_average'].median():.2f}/10")
        report.append(f"   • Highest Rating: {self.df_cleaned['vote_average'].max():.2f}/10")
        report.append(f"   • Lowest Rating: {self.df_cleaned['vote_average'].min():.2f}/10")

        report.append("\n2. RATING DISTRIBUTION")
        for category in ['Excellent', 'Good', 'Average', 'Poor']:
            count = (self.df_cleaned['rating_category'] == category).sum()
            percentage = (count / len(self.df_cleaned)) * 100
            report.append(f"   • {category}: {count} movies ({percentage:.1f}%)")

        report.append("\n3. POPULARITY INSIGHTS")
        report.append(f"   • Average Popularity: {self.df_cleaned['popularity'].mean():.2f}")
        report.append(f"   • Most Popular Movie: {self.df_cleaned.loc[self.df_cleaned['popularity'].idxmax(), 'title']}")
        report.append(f"     - Popularity Score: {self.df_cleaned['popularity'].max():.2f}")

        report.append("\n4. TOP 10 HIGHEST RATED MOVIES")
        top_10 = self.df_cleaned.nlargest(10, 'vote_average')[['title', 'vote_average', 'year']]
        for idx, (_, row) in enumerate(top_10.iterrows(), 1):
            report.append(f"   {idx:2d}. {row['title']} ({int(row['year'])}) - {row['vote_average']}/10")

        report.append("\n5. TEMPORAL TRENDS")
        report.append(f"   • Earliest Movie: {int(self.df_cleaned['year'].min())}")
        report.append(f"   • Latest Movie: {int(self.df_cleaned['year'].max())}")
        report.append(f"   • Most Productive Year: {int(self.df_cleaned['year'].mode()[0])}")

        year_avg = self.df_cleaned.groupby('year')['vote_average'].mean()
        best_year = year_avg.idxmax()
        report.append(f"   • Year with Highest Average Rating: {int(best_year)} ({year_avg.max():.2f}/10)")

        report.append("\n6. AUDIENCE ENGAGEMENT")
        report.append(f"   • Average Vote Count: {self.df_cleaned['vote_count'].mean():.0f}")
        report.append(f"   • Most Voted Movie: {self.df_cleaned.loc[self.df_cleaned['vote_count'].idxmax(), 'title']}")
        report.append(f"     - Vote Count: {self.df_cleaned['vote_count'].max():,.0f}")

        report.append("\n7. DATA QUALITY SUMMARY")
        report.append(f"   • Total Records: {len(self.df)}")
        report.append(f"   • Records After Cleaning: {len(self.df_cleaned)}")
        report.append(f"   • Records Removed: {len(self.df) - len(self.df_cleaned)}")
        report.append(f"   • Missing Values: {self.df_cleaned.isnull().sum().sum()}")
        report.append(f"   • Duplicate Records: 0 (removed)")

        report.append("\n" + "="*70)
        report.append("END OF REPORT")
        report.append("="*70)

        report_text = '\n'.join(report)
        with open('analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(report_text)
        print("\n✅ Report saved to: analysis_report.txt")

    def save_cleaned_data(self):
        print("\n" + "="*70)
        print("SAVING CLEANED DATA")
        print("="*70)

        self.df_cleaned.to_csv('movies_cleaned.csv', index=False)
        print("✅ Cleaned dataset saved to: movies_cleaned.csv")


def main():
    print("\n" + "="*70)
    print("TASK 1: DATA CLEANING & VISUALIZATION PROJECT")
    print("="*70)
    print("\nObjective: Work on a raw dataset to clean, process, and visualize insights")
    print("\nKey Features:")
    print("  ✓ Handle missing values, outliers, and duplicates")
    print("  ✓ Use Python libraries like Pandas, Matplotlib, Seaborn")
    print("  ✓ Create dashboards or visual reports of key findings")
    print("\n" + "="*70)

    analyzer = MovieDataAnalyzer(os.path.join('..', 'movies.csv'))

    analyzer.load_data()
    analyzer.explore_data()
    analyzer.clean_data()
    analyzer.create_visualizations()
    analyzer.generate_report()
    analyzer.save_cleaned_data()

    print("\n" + "="*70)
    print("✅ PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\n📁 Output Files Generated:")
    print("   1. movies_cleaned.csv - Full cleaned dataset")
    print("   2. analysis_report.txt - Comprehensive analysis report")
    print("   3. visualizations/ - 8 visualization files")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
