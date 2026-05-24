from config import azure_completion_client
from planner import SingleFilePlan

def write_markdown_document(file_plan, archive_title: str) -> str:
    """
    Generates high-fidelity file content based completely on generic 
    instructions provided by the planner.
    """
    from config import azure_completion_client

    writer_prompt = (
        f"You are an Expert Content Generator. Your task is to write the complete contents "
        f"for a file named '{file_plan.file_name}' under the broader topic/archive: '{archive_title}'.\n\n"
        f"Generation Instructions:\n"
        f"{file_plan.generation_instructions}\n\n"
        f"Requirements:\n"
        f"- Return only the raw file content. Do not wrap it in markdown block quotes (like ```markdown) or add chat conversations.\n"
        f"- Match the exact formatting structure, data style, or keys requested in the instructions above."
    )

    response = azure_completion_client(
        is_planner=False, # Call your writer/generation model deployment
        messages=[{"role": "user", "content": writer_prompt}]
    )

    return response.choices[0].message.content.strip()