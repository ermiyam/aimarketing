import jsonlines
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from pathlib import Path
import numpy as np

def load_combined_data():
    """Load the combined marketing learning database."""
    data = []
    with jsonlines.open('combined_learning_data/combined_marketing_learning_database.jsonl') as reader:
        for obj in reader:
            data.append(obj)
    return data

def analyze_failure_patterns(data):
    """Analyze patterns in marketing failures."""
    print("\nAnalyzing Marketing Failure Patterns...")
    
    # Extract all subtopics
    all_subtopics = []
    for item in data:
        if 'subtopics' in item:
            all_subtopics.extend(item['subtopics'])
    
    # Count occurrences of each failure type
    failure_counts = Counter(all_subtopics)
    
    # Group similar failures
    failure_categories = {
        'Targeting Issues': ['targeting', 'audience', 'demographic'],
        'Design Problems': ['design', 'visual', 'UI', 'UX'],
        'Research Gaps': ['research', 'customer research', 'market research'],
        'Platform Issues': ['platform', 'channel', 'medium'],
        'Technical Problems': ['mobile', 'technical', 'compatibility'],
        'Timing Issues': ['timing', 'schedule', 'seasonal'],
        'Budget Problems': ['budget', 'cost', 'financial'],
        'Branding Issues': ['branding', 'identity', 'copycat'],
        'Testing Gaps': ['testing', 'validation', 'experiment'],
        'Strategy Problems': ['strategy', 'planning', 'approach']
    }
    
    # Count failures by category
    category_counts = {category: 0 for category in failure_categories.keys()}
    for failure, count in failure_counts.items():
        for category, keywords in failure_categories.items():
            if any(keyword.lower() in failure.lower() for keyword in keywords):
                category_counts[category] += count
    
    return category_counts, failure_counts, failure_categories

def create_visualizations(category_counts, failure_counts, failure_categories):
    """Create visualizations of the analysis."""
    # Create output directory for visualizations
    output_dir = Path('analysis_visualizations')
    output_dir.mkdir(exist_ok=True)
    
    # 1. Bar chart of failure categories
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()))
    plt.title('Marketing Failure Categories Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Occurrences')
    plt.tight_layout()
    plt.savefig(output_dir / 'failure_categories.png')
    plt.close()
    
    # 2. Top 20 specific failures
    top_failures = dict(failure_counts.most_common(20))
    plt.figure(figsize=(12, 8))
    sns.barplot(x=list(top_failures.values()), y=list(top_failures.keys()))
    plt.title('Top 20 Most Common Marketing Failures')
    plt.xlabel('Number of Occurrences')
    plt.tight_layout()
    plt.savefig(output_dir / 'top_failures.png')
    plt.close()
    
    # 3. Pie chart of failure categories
    plt.figure(figsize=(10, 10))
    plt.pie(category_counts.values(), labels=category_counts.keys(), autopct='%1.1f%%')
    plt.title('Marketing Failure Categories Distribution')
    plt.tight_layout()
    plt.savefig(output_dir / 'failure_categories_pie.png')
    plt.close()
    
    # 4. Heatmap of failure correlations
    failure_matrix = np.zeros((len(failure_categories), len(failure_categories)))
    for i, (cat1, keywords1) in enumerate(failure_categories.items()):
        for j, (cat2, keywords2) in enumerate(failure_categories.items()):
            if i != j:
                # Count co-occurrences of failures from different categories
                co_occurrences = sum(1 for failure in failure_counts.keys() 
                                  if any(k1.lower() in failure.lower() for k1 in keywords1) and
                                     any(k2.lower() in failure.lower() for k2 in keywords2))
                failure_matrix[i, j] = co_occurrences
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(failure_matrix, annot=True, fmt='.0f', 
                xticklabels=failure_categories.keys(),
                yticklabels=failure_categories.keys(),
                cmap='YlOrRd')
    plt.title('Failure Category Correlations')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'failure_correlations.png')
    plt.close()

def generate_recommendations(category_counts, failure_counts):
    """Generate recommendations based on failure patterns."""
    recommendations = []
    
    # Overall recommendations
    recommendations.append("\nStrategic Recommendations:")
    
    # 1. Targeting and Strategy Focus
    if category_counts['Targeting Issues'] > sum(category_counts.values()) * 0.2:
        recommendations.append("""
1. Enhanced Audience Targeting:
   - Implement advanced audience segmentation
   - Use data-driven targeting strategies
   - Regularly update audience personas
   - Conduct A/B testing for targeting parameters""")
    
    if category_counts['Strategy Problems'] > sum(category_counts.values()) * 0.2:
        recommendations.append("""
2. Strategic Planning Improvements:
   - Develop comprehensive marketing strategies
   - Create detailed implementation plans
   - Set clear KPIs and success metrics
   - Regular strategy reviews and adjustments""")
    
    # 2. Technical and Design Recommendations
    if category_counts['Technical Problems'] > sum(category_counts.values()) * 0.05:
        recommendations.append("""
3. Technical Excellence:
   - Ensure mobile-first design approach
   - Regular technical audits
   - Cross-platform compatibility testing
   - Performance optimization""")
    
    if category_counts['Design Problems'] > sum(category_counts.values()) * 0.05:
        recommendations.append("""
4. Design Best Practices:
   - Implement user-centered design
   - Regular design reviews
   - A/B testing for design elements
   - Accessibility compliance""")
    
    # 3. Research and Testing Recommendations
    if category_counts['Research Gaps'] > sum(category_counts.values()) * 0.05:
        recommendations.append("""
5. Research Implementation:
   - Regular market research
   - Customer feedback collection
   - Competitor analysis
   - Data-driven decision making""")
    
    if category_counts['Testing Gaps'] > sum(category_counts.values()) * 0.05:
        recommendations.append("""
6. Testing Framework:
   - Implement continuous testing
   - A/B testing for all major changes
   - User testing sessions
   - Performance monitoring""")
    
    return recommendations

def generate_detailed_report(category_counts, failure_counts, recommendations):
    """Generate a detailed report of the analysis."""
    report = []
    
    # Overall statistics
    total_failures = sum(category_counts.values())
    report.append(f"\nTotal Marketing Failures Analyzed: {total_failures}")
    
    # Category breakdown
    report.append("\nFailure Categories Breakdown:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_failures) * 100
        report.append(f"{category}: {count} failures ({percentage:.1f}%)")
    
    # Top specific failures
    report.append("\nTop 10 Most Common Specific Failures:")
    for failure, count in failure_counts.most_common(10):
        report.append(f"- {failure}: {count} occurrences")
    
    # Add recommendations
    report.extend(recommendations)
    
    # Write report to file
    with open('analysis_visualizations/marketing_failures_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    return report

def main():
    print("Loading combined marketing database...")
    data = load_combined_data()
    
    print("Analyzing failure patterns...")
    category_counts, failure_counts, failure_categories = analyze_failure_patterns(data)
    
    print("Creating visualizations...")
    create_visualizations(category_counts, failure_counts, failure_categories)
    
    print("Generating recommendations...")
    recommendations = generate_recommendations(category_counts, failure_counts)
    
    print("Generating detailed report...")
    report = generate_detailed_report(category_counts, failure_counts, recommendations)
    
    # Print report to console
    print('\n'.join(report))
    print("\nAnalysis complete! Check the 'analysis_visualizations' directory for visualizations and detailed report.")

if __name__ == "__main__":
    main() 