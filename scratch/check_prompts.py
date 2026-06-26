"""
Script kiem tra prompt thuc te duoc tao ra tu wad_dataset.py.
Khong can load dataset that -- chi mock data dau vao de verify text.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")


# ── Mock samples ──────────────────────────────────────────────────────────────
SAMPLE_WITH_QA = {
    "frame_path": "some/path",
    "area_type": "Pedestrian Path",
    "weather_condition": "Sunny",
    "traffic_flow_rating": "High",
    "summary": "on a pedestrian street, there are pedestrians passing ahead.",
    "QA": {
        "Q": "can you tell me what is in front?",
        "A": "there is a billboard with a car advertisement in front of you."
    },
    "alter": None,
}

SAMPLE_ALTER_ONLY = {
    "frame_path": "some/path",
    "area_type": "Pedestrian Path",
    "weather_condition": "Sunny",
    "traffic_flow_rating": "High",
    "summary": "on a pedestrian street, there are pedestrians passing ahead.",
    "QA": None,
    "alter": "ahead there are pedestrians gathering. please slow down and avoid them towards 1 o'clock.",
}


# ── Copy chính xác logic từ wad_dataset.py __getitem__ ────────────────────────
def build_prompts(sample: dict, response_format: str):
    has_question = sample.get('QA') and sample['QA'].get('Q')

    if response_format == 'direct_text':
        if has_question:
            text_content = (
                "Based on this image, answer the following question for a visually impaired user directly in natural language.\n"
                f"Question: {sample['QA']['Q']}"
            )
        else:
            text_content = (
                "Describe the scene for a visually impaired user based on this image."
                "\nFocus on immediate obstacles, safe direction, and what action the user should take."
                "\nProvide only the final spoken guidance in natural language."
            )
    else:
        text_content = """

Analyze: location, weather, traffic, scene → then give instruction.

Follow Chain-of-Thought reasoning:
1. Perception: Extract "location", "weather", and "traffic".
2. Comprehension: Synthesize details into the "scene".
3. Decision: Formulate the final "instruction"."""

        if has_question:
            text_content += f"\n\nQuestion: {sample['QA']['Q']}"
            text_content += """\n\nFormat response:
<answer>{"location": "...", "weather": "...", "traffic": "...", "scene": "<concise visual summary, max 2 sentences>", "instruction": "<your answer to the question>"}</answer>"""
        else:
            text_content += """\n\nFormat response:
<answer>{"location": "...", "weather": "...", "traffic": "...", "scene": "<concise visual summary, max 2 sentences>", "instruction": "<actionable alert and guidance>"}</answer>"""

    question = f"<image>\n{text_content}"
    qformer_text = text_content.strip()
    return question, qformer_text


# -- In ket qua ----------------------------------------------------------------
def print_case(title: str, sample: dict, response_format: str):
    question, qformer_text = build_prompts(sample, response_format)
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {title}  |  response_format='{response_format}'")
    print(sep)
    print("\n[Q-Former nhan -- qformer_text]:")
    print("-" * 60)
    print(qformer_text)
    print("-" * 60)
    print("\n[LLM nhan -- question]:")
    print("-" * 60)
    print(question)
    print("-" * 60)


if __name__ == "__main__":
    for fmt in ["direct_text", "structured_json"]:
        print_case("CASE 1: Co cau hoi QA", SAMPLE_WITH_QA, fmt)
        print_case("CASE 2: Alter (khong co QA)", SAMPLE_ALTER_ONLY, fmt)
