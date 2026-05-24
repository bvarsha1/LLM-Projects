from pydantic import BaseModel, Field
from typing import List

# 📝 1. A Completely Generic File Layout
class SingleFilePlan(BaseModel):
    file_name: str = Field(..., description="The recommended file name ending in .md (e.g., data_log.md, documentation.md).")
    generation_instructions: str = Field(..., description="Specific instructions on what content, format, headers, or data points to generate for this specific file based on the user request.")

# 📦 2. A Completely Generic Document Archive Blueprint
class BatchEcosystemPlan(BaseModel):
    archive_title: str = Field(..., description="A short name or theme for this file generation batch.")
    summary: str = Field(..., description="A quick overview of what this collection of files accomplishes.")
    files_to_generate: List[SingleFilePlan] = Field(..., description="The dynamic array of 1 or more files that need to be generated to satisfy the prompt.")

def generate_specification(topic: str) -> BatchEcosystemPlan:
    """Calls Azure AI Foundry to dynamically plan out the required files based on user instruction."""
    from config import azure_completion_client
    
    planner_prompt = (
        f"You are an Advanced Document Generator Planner.\n"
        f"Analyze the user's request: '{topic}'\n\n"
        f"Instructions:\n"
        f"1. Determine how many files are needed to cleanly fulfill the user's request (could be 1 single file, or a collection of multiple files).\n"
        f"2. Plan the specific file names and precise layout/content requirements for each file as implied by the user's prompt.\n"
        f"3. Do not assume or inject any corporate schemas, company contexts, or strict layout requirements unless explicitly requested by the user."
    )
    
    response = azure_completion_client(
        is_planner=True,
        messages=[{"role": "user", "content": planner_prompt}],
        response_format=BatchEcosystemPlan
    )
    
    raw_content = response.choices[0].message.content
    return BatchEcosystemPlan.model_validate_json(raw_content)

def format_spec_to_table(spec: BatchEcosystemPlan) -> str:
    """Renders the blueprint layout cleanly using HTML breaks inside markdown cells."""
    md_table = f"### 📊 Generation Plan: {spec.archive_title}\n"
    md_table += f"> {spec.summary}\n\n"
    md_table += "| 📁 Target Output Filename | 📝 Generation Context and Directives |\n"
    md_table += "| :--- | :--- |\n"
    
    for file in spec.files_to_generate:
        # 1. Escaping any raw pipe characters so they don't break the column borders
        safe_instructions = file.generation_instructions.replace("|", "\\|")
        
        # 2. Converting standard code backticks so they don't conflict with our cell wrapper
        safe_instructions = safe_instructions.replace("```", "'''")
        
        # 3. Replacing real newlines with HTML line breaks to force print-like vertical spacing
        formatted_text = safe_instructions.replace("\n", "<br>")
        
        # 4. Constructing the row with the entire text completely preserved inside an HTML block
        md_table += f"| `{file.file_name}` | <div style='font-size: 0.9em; line-height: 1.4; font-family: monospace;'>{formatted_text}</div> |\n"
        
    return md_table