from autonomous_ai import AutonomousAI, Industry
import time
import json
import random

def get_industry_specific_queries(industry: Industry) -> list:
    """Get industry-specific marketing queries"""
    queries = {
        Industry.TECH: [
            "How can we leverage AI for customer segmentation?",
            "What are the best practices for SaaS product marketing?",
            "How do we optimize our tech product launch strategy?",
            "What metrics indicate successful tech product adoption?",
            "How can we improve our technical content marketing?",
            "What are the latest trends in tech customer experience?",
            "How do we measure tech product marketing ROI?",
            "What strategies work best for tech brand building?",
            "How can we enhance our tech social media presence?",
            "What are effective tech industry lead generation tactics?"
        ],
        Industry.FINANCE: [
            "How can we improve financial product adoption rates?",
            "What are the best practices for fintech marketing?",
            "How do we optimize investment product promotions?",
            "What metrics indicate successful financial service adoption?",
            "How can we enhance our financial content marketing?",
            "What are the latest trends in financial customer experience?",
            "How do we measure financial marketing ROI?",
            "What strategies work best for financial brand building?",
            "How can we improve our financial social media presence?",
            "What are effective financial industry lead generation tactics?"
        ],
        Industry.HEALTHCARE: [
            "How can we improve patient engagement rates?",
            "What are the best practices for healthcare digital marketing?",
            "How do we optimize medical service promotions?",
            "What metrics indicate successful healthcare service adoption?",
            "How can we enhance our healthcare content marketing?",
            "What are the latest trends in patient experience?",
            "How do we measure healthcare marketing ROI?",
            "What strategies work best for healthcare brand building?",
            "How can we improve our healthcare social media presence?",
            "What are effective healthcare industry lead generation tactics?"
        ],
        Industry.RETAIL: [
            "How can we improve customer retention rates?",
            "What are the best practices for retail digital marketing?",
            "How do we optimize retail promotions?",
            "What metrics indicate successful retail service adoption?",
            "How can we enhance our retail content marketing?",
            "What are the latest trends in retail customer experience?",
            "How do we measure retail marketing ROI?",
            "What strategies work best for retail brand building?",
            "How can we improve our retail social media presence?",
            "What are effective retail industry lead generation tactics?"
        ]
    }
    return queries.get(industry, [])

def get_cross_industry_queries() -> list:
    """Get queries that benefit from cross-industry knowledge"""
    return [
        "How can we improve customer engagement across channels?",
        "What are the best practices for digital transformation?",
        "How do we optimize our omnichannel marketing strategy?",
        "What metrics indicate successful digital adoption?",
        "How can we enhance our content marketing strategy?",
        "What are the latest trends in customer experience?",
        "How do we measure digital marketing ROI?",
        "What strategies work best for brand building?",
        "How can we improve our social media presence?",
        "What are effective lead generation tactics?"
    ]

def main():
    # Initialize autonomous AI with tech industry focus
    ai = AutonomousAI(industry=Industry.TECH)
    
    print("Starting enhanced autonomous AI learning process...")
    
    # Phase 1: Industry-specific learning
    print("\nPhase 1: Industry-specific Learning")
    industry_queries = get_industry_specific_queries(Industry.TECH)
    for query in industry_queries:
        print(f"\nUser: {query}")
        response = ai.process_input(query)
        print(f"AI: {response}")
        time.sleep(2)
    
    # Phase 2: Cross-industry knowledge transfer
    print("\nPhase 2: Cross-industry Knowledge Transfer")
    cross_industry_queries = get_cross_industry_queries()
    for query in cross_industry_queries:
        print(f"\nUser: {query}")
        response = ai.process_input(query)
        print(f"AI: {response}")
        time.sleep(2)
    
    # Phase 3: Advanced learning strategies
    print("\nPhase 3: Advanced Learning Strategies")
    advanced_queries = [
        "How can we combine insights from different industries to create innovative marketing strategies?",
        "What patterns emerge when comparing marketing approaches across industries?",
        "How can we adapt successful strategies from other industries to our context?",
        "What are the common challenges and solutions across different industries?",
        "How can we leverage cross-industry knowledge for competitive advantage?"
    ]
    for query in advanced_queries:
        print(f"\nUser: {query}")
        response = ai.process_input(query)
        print(f"AI: {response}")
        time.sleep(2)
    
    # Show final comprehensive learning status
    print("\nEnhanced autonomous AI learning process completed!")
    print("Final Learning Status:")
    status = ai.get_learning_status()
    print(json.dumps(status, indent=2))
    
    # Generate learning insights
    print("\nLearning Insights:")
    print(f"Total Queries Processed: {status['steps_completed']}")
    print(f"Industry Knowledge: {status['learning_metrics']['industry_expertise']:.2f}")
    print(f"Cross-industry Knowledge: {status['learning_metrics']['cross_industry_knowledge']:.2f}")
    print(f"Strategy Effectiveness: {status['learning_metrics']['strategy_effectiveness']:.2f}")
    
    # Show active learning strategies
    print("\nActive Learning Strategies:")
    for strategy in status['learning_strategies']:
        print(f"- {strategy}")

if __name__ == "__main__":
    main() 