import requests
import json

def run_model(query):
    # Change the URL to your running Flask endpoint
    url = "http://127.0.0.1:5000/cricket-cot"
    payload = {"query": query}
    response = requests.post(url, json=payload)
    return response.json().get("response", "")

# Evaluation dataset: 5+ samples
EVAL_DATASET = [
    {
        "query": "Who won the ICC Cricket World Cup in 2019?",
        "expected": "England"
    },
    {
        "query": "How many centuries has Virat Kohli scored in ODIs?",
        "expected": "Virat Kohli has scored more than 45 ODI centuries"
    },
    {
        "query": "When is the next India vs Australia match?",
        "expected": "date or schedule"
    },
    {
        "query": "Tell me a fun cricket trivia",
        "expected": "fun fact or trivia"
    },
    {
        "query": "Show me Rohit Sharma's highest individual ODI score",
        "expected": "264"
    }
]

# Judge prompt for evaluation
JUDGE_PROMPT = """
You are an impartial cricket expert. Compare the following model output to the expected answer. 
Consider correctness, completeness, relevance, and clarity. 
If the model output matches the expected answer in meaning, reply with 'PASS'. Otherwise, reply with 'FAIL' and explain why.

Expected: {expected}
Model Output: {output}
Result:
"""

def judge_output(expected, output):
    # For demonstration, a simple string match (replace with LLM call for real use)
    if expected.lower() in output.lower():
        return "PASS"
    return f"FAIL: Expected '{expected}', got '{output}'"


def run_evaluation():
    results = []
    for sample in EVAL_DATASET:
        query = sample["query"]
        expected = sample["expected"]
        output = run_model(query)
        judge_result = judge_output(expected, output)
        results.append({
            "query": query,
            "expected": expected,
            "output": output,
            "judge": judge_result
        })
        print(f"Query: {query}\nExpected: {expected}\nOutput: {output}\nJudge: {judge_result}\n{'-'*40}")
    return results

if __name__ == "__main__":
    run_evaluation()
