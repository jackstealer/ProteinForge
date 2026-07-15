import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 6)


class YouTubeEDA:

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.numeric_cols = []
        self.categorical_cols = []

    def load_data(self):
        print("="*80)
        print("LOADING GLOBAL YOUTUBE STATISTICS DATA")
        print("="*80)
        self.df = pd.read_csv(self.filepath, encoding='latin-1')
        print(f"✓ Dataset loaded successfully!")
        print(f"  Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"\nFirst few column names:")
        print(f"  {', '.join(self.df.columns.tolist()[:10])}...")
        return self.df

    def initial_exploration(self):
        print("\n" + "="*80)
        print("INITIAL DATA EXPLORATION")
        print("="*80)

        print("\n📊 Dataset Shape:")
        print(f"   Rows: {self.df.shape[0]:,}")
        print(f"   Columns: {self.df.shape[1]}")

        print("\n📊 First 5 rows:")
        print(self.df.head())

        print("\n📊 Data Types:")
        print(self.df.dtypes)

        print("\n📊 Basic Statistics:")
        print(self.df.describe())

        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        print(f"\n📊 Column Types:")
        print(f"   Numeric columns: {len(self.numeric_cols)}")
        print(f"   Categorical columns: {len(self.categorical_cols)}")

    def data_quality_check(self):
        print("\n" + "="*80)
        print("DATA QUALITY ASSESSMENT")
        print("="*80)

        print("\n🔍 Missing Values Analysis:")
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)

        if len(missing_df) > 0:
            print(f"   Total columns with missing values: {len(missing_df)}")
            print(f"\n   Top 10 columns with most missing values:")
            print(missing_df.head(10).to_string(index=False))
        else:
            print("   ✓ No missing values found!")

        print("\n🔍 Duplicate Rows:")
        duplicates = self.df.duplicated().sum()
        print(f"   Duplicate rows: {duplicates}")

        print("\n🔍 Data Type Summary:")
        print(f"   Integer columns: {len(self.df.select_dtypes(include=['int64']).columns)}")
        print(f"   Float columns: {len(self.df.select_dtypes(include=['float64']).columns)}")
        print(f"   Object columns: {len(self.df.select_dtypes(include=['object']).columns)}")

        return missing_df

    def univariate_analysis(self):
        print("\n" + "="*80)
        print("UNIVARIATE ANALYSIS")
        print("="*80)

        if 'subscribers' in self.df.columns:
            print("\n📊 Subscribers Analysis:")
            print(f"   Mean: {self.df['subscribers'].mean():,.0f}")
            print(f"   Median: {self.df['subscribers'].median():,.0f}")
            print(f"   Max: {self.df['subscribers'].max():,.0f}")
            print(f"   Min: {self.df['subscribers'].min():,.0f}")
            print(f"   Std: {self.df['subscribers'].std():,.0f}")

        if 'video views' in self.df.columns:
            print("\n📊 Video Views Analysis:")
            print(f"   Mean: {self.df['video views'].mean():,.0f}")
            print(f"   Median: {self.df['video views'].median():,.0f}")
            print(f"   Max: {self.df['video views'].max():,.0f}")
            print(f"   Min: {self.df['video views'].min():,.0f}")

        if 'category' in self.df.columns:
            print("\n📊 Category Distribution:")
            category_counts = self.df['category'].value_counts().head(10)
            for cat, count in category_counts.items():
                print(f"   {cat}: {count} channels")

        if 'Country' in self.df.columns:
            print("\n📊 Top 10 Countries by Number of Channels:")
            country_counts = self.df['Country'].value_counts().head(10)
            for country, count in country_counts.items():
                print(f"   {country}: {count} channels")

    def bivariate_analysis(self):
        print("\n" + "="*80)
        print("BIVARIATE ANALYSIS")
        print("="*80)

        print("\n📊 Correlation Analysis:")
        key_numeric = ['subscribers', 'video views', 'uploads', 'video_views_for_the_last_30_days']
        available_cols = [col for col in key_numeric if col in self.df.columns]

        if len(available_cols) >= 2:
            corr_matrix = self.df[available_cols].corr()
            print("\n   Correlation Matrix:")
            print(corr_matrix)

            print("\n   Strongest Correlations:")
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
            corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            for col1, col2, corr_val in corr_pairs[:5]:
                print(f"   {col1} <-> {col2}: {corr_val:.4f}")

        if 'category' in self.df.columns and 'subscribers' in self.df.columns:
            print("\n📊 Average Subscribers by Category (Top 10):")
            cat_subs = self.df.groupby('category')['subscribers'].mean().sort_values(ascending=False).head(10)
            for cat, avg_subs in cat_subs.items():
                print(f"   {cat}: {avg_subs:,.0f}")

        if 'Country' in self.df.columns:
            print("\n📊 Channel Distribution by Country (Top 10):")
            country_dist = self.df['Country'].value_counts().head(10)
            for country, count in country_dist.items():
                print(f"   {country}: {count} channels")

    def identify_outliers(self):
        print("\n" + "="*80)
        print("OUTLIER DETECTION")
        print("="*80)

        key_cols = ['subscribers', 'video views', 'uploads']
        available_cols = [col for col in key_cols if col in self.df.columns]

        for col in available_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
            outlier_percent = (outliers / len(self.df)) * 100

            print(f"\n📊 {col}:")
            print(f"   Outliers detected: {outliers} ({outlier_percent:.1f}%)")
            print(f"   Range: [{self.df[col].min():,.0f}, {self.df[col].max():,.0f}]")
            print(f"   IQR bounds: [{lower_bound:,.0f}, {upper_bound:,.0f}]")

    def create_visualizations(self):
        print("\n" + "="*80)
        print("CREATING VISUALIZATIONS")
        print("="*80)

        if not os.path.exists('visualizations'):
            os.makedirs('visualizations')

        print("\n📈 Creating visualization 1/10: Subscribers Distribution...")
        if 'subscribers' in self.df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            axes[0].hist(self.df['subscribers'].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            axes[0].set_title('Distribution of Subscribers', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Subscribers')
            axes[0].set_ylabel('Frequency')
            axes[0].grid(alpha=0.3)

            axes[1].boxplot(self.df['subscribers'].dropna(), vert=True)
            axes[1].set_title('Subscribers Boxplot', fontsize=14, fontweight='bold')
            axes[1].set_ylabel('Subscribers')
            axes[1].grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig('visualizations/1_subscribers_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("📈 Creating visualization 2/10: Top 20 YouTubers...")
        if 'Youtuber' in self.df.columns and 'subscribers' in self.df.columns:
            top_20 = self.df.nlargest(20, 'subscribers')[['Youtuber', 'subscribers']]

            plt.figure(figsize=(14, 10))
            plt.barh(range(len(top_20)), top_20['subscribers'], color='coral', alpha=0.8)
            plt.yticks(range(len(top_20)), top_20['Youtuber'])
            plt.xlabel('Subscribers', fontsize=12)
            plt.title('Top 20 YouTubers by Subscribers', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig('visualizations/2_top_20_youtubers.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("📈 Creating visualization 3/10: Category Distribution...")
        if 'category' in self.df.columns:
            plt.figure(figsize=(12, 8))
            category_counts = self.df['category'].value_counts().head(15)
            plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90)
            plt.title('Distribution of Channels by Category (Top 15)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('visualizations/3_category_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("📈 Creating visualization 4/10: Country Distribution...")
        if 'Country' in self.df.columns:
            plt.figure(figsize=(14, 6))
            country_counts = self.df['Country'].value_counts().head(15)
            plt.bar(range(len(country_counts)), country_counts.values, color='mediumseagreen', alpha=0.7)
            plt.xticks(range(len(country_counts)), country_counts.index, rotation=45, ha='right')
            plt.ylabel('Number of Channels', fontsize=12)
            plt.title('Top 15 Countries by Number of YouTube Channels', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('visualizations/4_country_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("📈 Creating visualization 5/10: Subscribers vs Video Views...")
        if 'subscribers' in self.df.columns and 'video views' in self.df.columns:
            plt.figure(figsize=(12, 8))
            plt.scatter(self.df['subscribers'], self.df['video views'], alpha=0.5, c='purple', s=50)
            plt.xlabel('Subscribers', fontsize=12)
            plt.ylabel('Video Views', fontsize=12)
            plt.title('Subscribers vs Video Views', fontsize=14, fontweight='bold')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig('visualizations/5_subscribers_vs_views.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("📈 Creating visualization 6/10: Correlation Heatmap...")
        key_cols = ['subscribers', 'video views', 'uploads', 'video_views_for_the_last_30_days']
        available_cols = [col for col in key_cols if col in self.df.columns]

        if len(available_cols) >= 2:
            plt.figure(figsize=(10, 8))
            corr_matrix = self.df[available_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=1, fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title('Correlation Heatmap of Key Metrics', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig('visualizations/6_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("📈 Creating visualization 7/10: Avg Subscribers by Category...")
        if 'category' in self.df.columns and 'subscribers' in self.df.columns:
            plt.figure(figsize=(12, 8))
            cat_subs = self.df.groupby('category')['subscribers'].mean().sort_values(ascending=False).head(15)
            plt.barh(range(len(cat_subs)), cat_subs.values, color='teal', alpha=0.7)
            plt.yticks(range(len(cat_subs)), cat_subs.index)
            plt.xlabel('Average Subscribers', fontsize=12)
            plt.title('Average Subscribers by Category (Top 15)', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig('visualizations/7_avg_subscribers_category.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("📈 Creating visualization 8/10: Upload Count Distribution...")
        if 'uploads' in self.df.columns:
            plt.figure(figsize=(12, 6))
            plt.hist(self.df['uploads'].dropna(), bins=50, color='orange', edgecolor='black', alpha=0.7)
            plt.xlabel('Number of Uploads', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.title('Distribution of Upload Counts', fontsize=14, fontweight='bold')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig('visualizations/8_uploads_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("📈 Creating visualization 9/10: Channel Type Distribution...")
        if 'channel_type' in self.df.columns:
            plt.figure(figsize=(12, 8))
            channel_counts = self.df['channel_type'].value_counts().head(12)
            plt.bar(range(len(channel_counts)), channel_counts.values, color='crimson', alpha=0.7)
            plt.xticks(range(len(channel_counts)), channel_counts.index, rotation=45, ha='right')
            plt.ylabel('Number of Channels', fontsize=12)
            plt.title('Channel Type Distribution (Top 12)', fontsize=14, fontweight='bold')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('visualizations/9_channel_type_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("📈 Creating visualization 10/10: Channel Creation Timeline...")
        if 'created_year' in self.df.columns:
            plt.figure(figsize=(14, 6))
            year_counts = self.df['created_year'].value_counts().sort_index()
            plt.plot(year_counts.index, year_counts.values, marker='o', linewidth=2,
                    markersize=6, color='navy')
            plt.xlabel('Year', fontsize=12)
            plt.ylabel('Number of Channels Created', fontsize=12)
            plt.title('YouTube Channels Created Over Time', fontsize=14, fontweight='bold')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig('visualizations/10_creation_timeline.png', dpi=300, bbox_inches='tight')
            plt.close()

        print("\n✅ All visualizations created successfully!")
        print("   Saved in: visualizations/")

    def generate_insights_report(self):
        print("\n" + "="*80)
        print("GENERATING INSIGHTS REPORT")
        print("="*80)

        report = []
        report.append("="*80)
        report.append("GLOBAL YOUTUBE STATISTICS - EDA REPORT")
        report.append("="*80)
        report.append(f"\nGenerated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"\nDataset: Global YouTube Statistics.csv")
        report.append(f"Total Records: {len(self.df):,}")
        report.append(f"Total Features: {self.df.shape[1]}")

        report.append("\n" + "-"*80)
        report.append("KEY INSIGHTS AND PATTERNS")
        report.append("-"*80)

        report.append("\n1. OVERALL YOUTUBE LANDSCAPE")
        report.append(f"   • Total YouTubers analyzed: {len(self.df):,}")
        if 'subscribers' in self.df.columns:
            report.append(f"   • Average subscribers: {self.df['subscribers'].mean():,.0f}")
            report.append(f"   • Median subscribers: {self.df['subscribers'].median():,.0f}")
            report.append(f"   • Top channel: {self.df.loc[self.df['subscribers'].idxmax(), 'Youtuber']} " +
                        f"with {self.df['subscribers'].max():,.0f} subscribers")

        if 'category' in self.df.columns:
            report.append("\n2. CATEGORY ANALYSIS")
            report.append(f"   • Total categories: {self.df['category'].nunique()}")
            top_category = self.df['category'].value_counts().index[0]
            top_category_count = self.df['category'].value_counts().values[0]
            report.append(f"   • Most popular category: {top_category} ({top_category_count} channels)")

            if 'subscribers' in self.df.columns:
                top_earning_cat = self.df.groupby('category')['subscribers'].mean().sort_values(ascending=False).index[0]
                report.append(f"   • Highest avg subscribers: {top_earning_cat}")

        if 'Country' in self.df.columns:
            report.append("\n3. GEOGRAPHIC DISTRIBUTION")
            report.append(f"   • Countries represented: {self.df['Country'].nunique()}")
            top_country = self.df['Country'].value_counts().index[0]
            top_country_count = self.df['Country'].value_counts().values[0]
            report.append(f"   • Top country: {top_country} ({top_country_count} channels)")
            report.append(f"   • Top 5 countries:")
            for i, (country, count) in enumerate(self.df['Country'].value_counts().head(5).items(), 1):
                report.append(f"     {i}. {country}: {count} channels")

        if 'uploads' in self.df.columns:
            report.append("\n4. CONTENT PRODUCTION")
            report.append(f"   • Average uploads: {self.df['uploads'].mean():,.0f}")
            report.append(f"   • Median uploads: {self.df['uploads'].median():,.0f}")
            report.append(f"   • Max uploads: {self.df['uploads'].max():,.0f}")
            most_prolific = self.df.loc[self.df['uploads'].idxmax(), 'Youtuber']
            report.append(f"   • Most prolific channel: {most_prolific}")

        if 'subscribers' in self.df.columns and 'video views' in self.df.columns:
            corr = self.df['subscribers'].corr(self.df['video views'])
            report.append("\n5. KEY CORRELATIONS")
            report.append(f"   • Subscribers vs Video Views correlation: {corr:.4f}")
            if corr > 0.7:
                report.append("     → Strong positive correlation detected")
            elif corr > 0.4:
                report.append("     → Moderate positive correlation detected")
            else:
                report.append("     → Weak correlation detected")

        report.append("\n6. DATA QUALITY SUMMARY")
        missing_count = self.df.isnull().sum().sum()
        report.append(f"   • Total missing values: {missing_count:,}")
        report.append(f"   • Duplicate records: {self.df.duplicated().sum()}")

        report.append("\n" + "="*80)
        report.append("END OF REPORT")
        report.append("="*80)

        report_text = '\n'.join(report)
        with open('eda_insights_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(report_text)
        print("\n✅ Report saved to: eda_insights_report.txt")


def main():
    print("\n" + "="*80)
    print("TASK 3: EXPLORATORY DATA ANALYSIS (EDA) PROJECT")
    print("="*80)
    print("\nObjective: Analyze dataset to uncover patterns and trends")
    print("\nKey Features:")
    print("  ✓ Use statistical summaries and visualizations")
    print("  ✓ Identify correlations and key influencing factors")
    print("  ✓ Present insights in a structured report")
    print("\n" + "="*80)

    eda = YouTubeEDA('Global YouTube Statistics.csv')

    eda.load_data()
    eda.initial_exploration()
    eda.data_quality_check()
    eda.univariate_analysis()
    eda.bivariate_analysis()
    eda.identify_outliers()
    eda.create_visualizations()
    eda.generate_insights_report()

    print("\n" + "="*80)
    print("✅ EDA PROJECT COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("\n📁 Output Files Generated:")
    print("   1. eda_insights_report.txt - Comprehensive insights report")
    print("   2. visualizations/ - 10 visualization files")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
