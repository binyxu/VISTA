import re


def extract_final_answer(response: str, mcq_mode: bool = False) -> str:
    """Extract the final answer from model response.

    Args:
        response: Raw model response
        mcq_mode: When True (multiple-choice), keep only the first line after
            the marker (expected to contain just the letter(s) like "(A)(B)").
            When False (open-ended), preserve the full multi-line answer.

    Returns:
        Extracted answer string
    """
    # Strip thinking blocks (e.g. from Qwen3 thinking mode)
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()

    # Try to extract answer after "##Answer:" marker
    if "##Answer:" in response:
        parts = response.split("##Answer:", 1)
        if len(parts) > 1:
            answer = parts[1].strip()
            if mcq_mode:
                # MCQ answers are a single letter/combination — first line only.
                answer = answer.split('\n')[0].strip()
            return answer

    # If no marker found, return the original response
    return response.strip()
