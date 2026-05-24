import os
import shutil
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import planner
import writer
from app import build_interface

def worker_generate_file(file_plan, archive_title, output_dir_name, idx, total_files):
    """
    Isolated worker task executed concurrently inside the ThreadPool.
    Handles generating and writing a single file to disk.
    """
    try:
        # Call the writer model deployment endpoint with generic topic context
        markdown_content = writer.write_markdown_document(file_plan, archive_title)
        
        # Enforce clean naming structures
        target_file_name = file_plan.file_name.strip()
        
        # Write asset directly into the target workspace path
        target_filepath = os.path.join(output_dir_name, target_file_name)
        
        # Ensure any nested subfolders planned by the model are created automatically
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        
        with open(target_filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)
            
        return f"✅ **[{idx}/{total_files}]** Successfully compiled `{target_file_name}`"
    except Exception as e:
        return f"❌ **[{idx}/{total_files}]** Failed to build `{file_plan.file_name}`: *{str(e)}*"

def execute_pipeline(topic: str) -> tuple:
    """Coordinates parallel pool execution and compiles the final zip payload."""
    try:
        # Step 1: Run Generic Planner Phase to map out files based entirely on user intent
        generation_plan = planner.generate_specification(topic)
        formatted_table = planner.format_spec_to_table(generation_plan)
    except Exception as e:
        return f"### ❌ Planner Error\n```text\n{str(e)}\n```", "Pipeline execution aborted during planning phase.", None

    # Step 2: Establish an abstract temporary workspace directory using a clean slug
    # Using alphanumeric sanitization to support topics that aren't strict business names
    clean_slug = re.sub(r'[^a-zA-Z0-9_]', '', generation_plan.archive_title.lower().replace(" ", "_"))
    output_dir_name = f"synthetic_{clean_slug}_data"
    
    if os.path.exists(output_dir_name):
        shutil.rmtree(output_dir_name)
    os.makedirs(output_dir_name)

    total_files = len(generation_plan.files_to_generate)
    generation_results_log = []

    # Step 3: Initialize the ThreadPoolExecutor for parallel execution
    max_workers = min(8, total_files)
    
    print(f"🧵 Spin-up ThreadPool: Processing {total_files} assets with {max_workers} parallel workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                worker_generate_file, 
                file_plan, 
                generation_plan.archive_title, 
                output_dir_name, 
                idx, 
                total_files
            )
            for idx, file_plan in enumerate(generation_plan.files_to_generate, 1)
        ]
        
        for future in as_completed(futures):
            log_line = future.result()
            generation_results_log.append(log_line)
            print(log_line)

    generation_log_md = "### ✒️ Parallel Worker Log Summary\n---\n"
    generation_log_md += "\n\n".join(generation_results_log)

    # Step 4: Compress the populated directory into a zip archive
    try:
        zip_archive_base = f"{output_dir_name}_package"
        compiled_zip_path = shutil.make_archive(zip_archive_base, 'zip', output_dir_name)
        shutil.rmtree(output_dir_name) 
        
        generation_log_md += f"\n\n📦 **Dataset Archive Compiled Successfully!**\nAll assets bundled into .zip payload."
    except Exception as zip_err:
        return formatted_table, f"### ❌ Zip Compilation Failure\n```text\n{str(zip_err)}\n```", None

    return formatted_table, generation_log_md, compiled_zip_path

if __name__ == "__main__":
    ui = build_interface(pipeline_callback_fn=execute_pipeline)
    ui.launch()