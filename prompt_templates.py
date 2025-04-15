"""
Prompt templates for ReSearch-style training and inference.
"""

from typing import Optional, List, Dict
import re

def format_qa_with_reasoning_tags(question: str, answer: str, context: Optional[str] = None) -> str:
    """
    Formats a QA pair into the reasoning+search format for ReSearch training.
    
    Args:
        question: The input question
        answer: The target answer
        context: Optional context to include
        
    Returns:
        str: Formatted prompt with reasoning tags
    """
    prompt = f"""<question> {question.strip()} </question>

<think> Let's break this down step by step. </think>
<search> [insert first search query here based on question] </search>
<result> [insert first result here] </result>
<think> Using that, let's keep reasoning... </think>
<search> [insert second query if needed] </search>
<result> [insert second result if needed] </result>
<answer> \\boxed{{{answer.strip()}}} </answer>
"""
    return prompt

def format_inference_prompt(question: str) -> str:
    """
    Formats a question-only prompt for inference-time use.
    
    Args:
        question: The input question
        
    Returns:
        str: Formatted inference prompt
    """
    return f"""<question> {question.strip()} </question>

<think> Let's break this down step by step. </think>
"""

def insert_search_result(generated_text: str, search_query: str, search_result: str) -> str:
    """
    After <search> is found, inject <result>...</result> using retrieved text.
    
    Args:
        generated_text: The current generated text
        search_query: The search query that was used
        search_result: The retrieved search result
        
    Returns:
        str: Updated text with search result inserted
    """
    search_tag = f"<search>{search_query}</search>"
    result_tag = f"{search_tag}<result>{search_result}</result>"
    return generated_text.replace(search_tag, result_tag)

def extract_search_queries(text: str) -> List[str]:
    """
    Extract search queries from generated text.
    
    Args:
        text: The generated text containing search tags
        
    Returns:
        List[str]: List of extracted search queries
    """
    pattern = r"<search>(.*?)</search>"
    return re.findall(pattern, text)

def extract_search_results(text: str) -> List[str]:
    """
    Extract search results from generated text.
    
    Args:
        text: The generated text containing result tags
        
    Returns:
        List[str]: List of extracted search results
    """
    pattern = r"<result>(.*?)</result>"
    return re.findall(pattern, text)

def extract_answer(text: str) -> str:
    """
    Extract the final answer from generated text.
    
    Args:
        text: The generated text containing answer tags
        
    Returns:
        str: The extracted answer
    """
    pattern = r"<answer>(.*?)</answer>"
    matches = re.findall(pattern, text)
    return matches[0] if matches else ""

def format_multi_hop_prompt(question: str, 
                          previous_results: List[str],
                          current_step: int,
                          total_steps: int) -> str:
    """
    Format a prompt for multi-hop reasoning.
    
    Args:
        question: The original question
        previous_results: Results from previous search steps
        current_step: Current step number
        total_steps: Total number of steps
        
    Returns:
        str: Formatted multi-hop prompt
    """
    prompt = f"""<question> {question.strip()} </question>

<think> Let's break this down step by step. </think>
"""
    
    # Add previous results
    for i, result in enumerate(previous_results):
        prompt += f"""<search> [previous query {i+1}] </search>
<result> {result} </result>
<think> Using that information... </think>
"""
    
    # Add current step
    prompt += f"""<search> [step {current_step} of {total_steps}] </search>
<result> [insert result here] </result>
"""
    
    return prompt

def format_reflexive_prompt(question: str,
                          current_answer: str,
                          feedback: str) -> str:
    """
    Format a prompt for reflexive reasoning and self-correction.
    
    Args:
        question: The original question
        current_answer: The current answer
        feedback: Feedback on the current answer
        
    Returns:
        str: Formatted reflexive prompt
    """
    return f"""<question> {question.strip()} </question>

<think> Let's analyze the current answer and feedback. </think>
<current_answer> {current_answer} </current_answer>
<feedback> {feedback} </feedback>
<think> Based on this feedback, let's improve the answer. </think>
<search> [insert search query for improvement] </search>
<result> [insert result here] </result>
<answer> [insert improved answer here] </answer>
""" 