import json
import os
from pathlib import Path
import gradio as gr
import litellm
from engine import SyntheticDataPipeline

# Initialize the modular processing pipeline engine
pipeline = SyntheticDataPipeline()

# Sample structural baseline template
default_config = {
  "volume": 5,
  "requirements": "Generate realistic enterprise customer profiles. Maintain global regional parity and avoid gender or age demographic skews.",
  "schema": {
    "account_id": {"type": "integer", "description": "Unique auto-incrementing customer primary key sequence identifier."},
    "email": {"type": "string", "description": "Lowercase verified corporate contact parameter address match."},
    "metrics_telemetry": {
      "type": "object",
      "description": "Nested consumer lifecycle engagement tracking telemetry metrics data block.",
      "properties": {
        "monthly_spend_usd": {"type": "number", "description": "Accurate floating-point transaction rate record metrics calculation."},
        "is_active": {"type": "boolean", "description": "True if consumer performed platform actions within 30 days."}
      }
    }
  }
}

# Styling for layouts
custom_css = """
.app-container { max-width: 1200px; margin: auto; padding: 1rem 2rem; }
.main-header { margin-bottom: 2rem; text-align: left; border-bottom: 1px solid #e2e8f0; padding-bottom: 1rem; }
.control-header { font-size: 1.1rem; font-weight: 600; margin-top: 1rem; margin-bottom: 0.5rem; color: #1a202c; }
"""

with gr.Blocks() as demo:
    
    with gr.Column(elem_classes="app-container"):
        
        # Un-styled Group keeping headers neat without background cards
        with gr.Group(elem_classes="main-header"):
            gr.Markdown("# 🧬 Universal Pydantic JSONL Synthetic Generator Workspace")
            gr.Markdown("Configure target generation layout parameters, inspect schema compilation layers, and evaluate outputs using local or cloud LLMs securely.")
        
        with gr.Row():
            # LEFT CONFIGURATION COLUMN
            with gr.Column(scale=1):
                
                gr.Markdown("### ⚙️ Primary Generation Configuration", elem_classes="control-header")
                gen_model = gr.Textbox(value="openai/gpt-4o-mini", placeholder="e.g., ollama/llama3.2, anthropic/claude-3-5-sonnet", label="Generation Target Model Name")
                gen_base_url = gr.Textbox(value="http://localhost:11434", placeholder="Default: http://localhost:11434", label="Generation Server API Base URL Link")
                gen_api_key = gr.Textbox(type="password", placeholder="Leave blank if local unauthenticated engine instance", label="Generation Endpoint API Key Credential")
                
                gr.Markdown("### ⚖️ Post-Generation Quality Audit Settings", elem_classes="control-header")
                use_judge_cb = gr.Checkbox(value=False, label="Activate Stage 2 Evaluation Judge Compliance Review Loop")
                judge_model = gr.Textbox(value="openai/gpt-4o-mini", placeholder="e.g., ollama/llama3.2, openai/gpt-4o", label="Judge Target Model Name")
                judge_base_url = gr.Textbox(value="http://localhost:11434", placeholder="Default: http://localhost:11434", label="Judge Server API Base URL Link")
                judge_api_key = gr.Textbox(type="password", placeholder="Leave blank if local unauthenticated engine instance", label="Judge Endpoint API Key Credential")

                gr.Markdown("### 📋 Generation Requirements & Structural JSON Schema", elem_classes="control-header")
                config_input = gr.Textbox(value=json.dumps(default_config, indent=2), lines=14, label="System Blueprint Specification Rules & Fields Definition Layout (JSON)")
                
                with gr.Row():
                    inspect_btn = gr.Button("🔍 Verify & Inspect Schema", variant="secondary")
                    run_btn = gr.Button("🚀 Run Processing Engine", variant="primary")
            
            # RIGHT OUTPUT & EVALUATION PREVIEW COLUMN
            with gr.Column(scale=1):
                
                gr.Markdown("### 🛡️ Runtime Engine Compilation Inspection Preview Matrix", elem_classes="control-header")
                schema_preview = gr.JSON(label="Live Generated Pydantic Class Validation Spec View Layout")
                
                gr.Markdown("### 📊 Live Generated Dataset Preview (First 5 Rows Out of File Stream)", elem_classes="control-header")
                output_preview = gr.JSON(label="Interactive JSONL Row Buffer View")
                
                gr.Markdown("#### 📦 Downstream Export Package Artifact Destination Link")
                file_downloader = gr.File(label="Download Generated .jsonl Dataset Output File Asset", interactive=False)
                
                gr.Markdown("### 🔬 Stage 2 Quality Assurance Evaluation Analytics Metric Logs", elem_classes="control-header")
                audit_view = gr.JSON(label="Independent Judge Critique Compliance Verification Report Payload")
                
                # FULLY INTEGRATED RECURSIVE REGENERATION ACTION BUTTON
                with gr.Row():
                    refine_btn = gr.Button("🔄 Refine & Regenerate (Apply Judge Feedback Loop)", variant="stop", visible=False)
                
                status_view = gr.Textbox(label="Operational System Context Engine Trace Processing Footprint Lines", interactive=False)

        # ----------------------------------------------------------------------
        # SYSTEM INTERACTIVE LOGIC CONNECTIONS WIRINGS
        # ----------------------------------------------------------------------
        
        # Action Callback 1: Check validation patterns without burning LLM API usage tokens
        def preview_schema_logic(config_str):
            try:
                cfg = json.loads(config_str)
                pydantic_model = pipeline.compile_schema_to_pydantic(cfg.get("schema", {}))
                return pydantic_model.model_json_schema(), "Configuration compiled successfully: Data structures properly mapped into runtime standard type models."
            except Exception as e:
                return {}, f"Configuration Exception detected while compiling properties: {str(e)}"

        inspect_btn.click(fn=preview_schema_logic, inputs=config_input, outputs=[schema_preview, status_view])

        # Core Helper: Handles local vs cloud endpoint routing configs dynamically 
        def configure_environment_keys(model, base_url, api_key):
            if "ollama" in model:
                litellm.api_base = base_url.strip() if base_url.strip() else "http://localhost:11434"
                os.environ["OPENAI_API_KEY"] = "bypassed"
            else:
                litellm.api_base = None
                if api_key.strip():
                    os.environ["OPENAI_API_KEY"] = api_key.strip()

        # Helper: Safely reads records stream straight from local file assets storage target 
        def extract_file_sample(filepath):
            extracted_lines_sample = []
            if Path(filepath).exists():
                with open(filepath, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= 5:
                            break
                        if line.strip():
                            extracted_lines_sample.append(json.loads(line.strip()))
            return extracted_lines_sample

        # Action Callback 2: Standard Engine Pipeline Run Block Execution
        def execution_pipeline_logic(gen_mod, gen_url, gen_key, process_judge, jd_mod, jd_url, jd_key, config_str):
            output_filepath = "spaces_dataset_output.jsonl"
            try:
                cfg = json.loads(config_str)
                vol = int(cfg.get("volume", 5))
                reqs = cfg.get("requirements", "")
                sch = cfg.get("schema", {})
                
                configure_environment_keys(gen_mod, gen_url, gen_key)
                records = pipeline.run_generation(gen_mod, reqs, sch, vol, output_filepath)
                sample_rows = extract_file_sample(output_filepath)
                
                if process_judge:
                    configure_environment_keys(jd_mod, jd_url, jd_key)
                    audit_payload = pipeline.run_post_audit(jd_mod, reqs, sch, records)
                    
                    show_refine = gr.update(visible=True)
                    return sample_rows, output_filepath, audit_payload, show_refine, f"Success: Created {len(records)} entries. Review critique notes and refine if corrections are necessary."
                
                return sample_rows, output_filepath, {"status": "Audit skipped."}, gr.update(visible=False), f"Success: Stored {len(records)} fresh records cleanly."
            except Exception as e:
                return [], None, {}, gr.update(visible=False), f"Execution Error triggered: {str(e)}"

        run_btn.click(
            fn=execution_pipeline_logic, 
            inputs=[gen_model, gen_base_url, gen_api_key, use_judge_cb, judge_model, judge_base_url, judge_api_key, config_input], 
            outputs=[output_preview, file_downloader, audit_view, refine_btn, status_view]
        )

        # Action Callback 3: Circular Feedback Loop Execution (Generates + Immediately Re-Audits with Judge)
        def refinement_loop_logic(gen_mod, gen_url, gen_key, process_judge, jd_mod, jd_url, jd_key, audit_report, config_str):
            output_filepath = "spaces_dataset_output.jsonl"
            try:
                cfg = json.loads(config_str)
                vol = int(cfg.get("volume", 5))
                original_reqs = cfg.get("requirements", "")
                sch = cfg.get("schema", {})
                
                # Turn previous audit run errors into instruction modifiers
                feedback_critique_summary = json.dumps(audit_report)
                compounded_requirements = (
                    f"{original_reqs}\n\n"
                    f"[CRITICAL REFINEMENT DIRECTIVE]: A prior verification audit flagged consistency issues. "
                    f"You must strictly adjust generation parameters to correct these errors: {feedback_critique_summary}"
                )
                
                # Step A: Re-generate records with combined error correction payload strings
                configure_environment_keys(gen_mod, gen_url, gen_key)
                records = pipeline.run_generation(gen_mod, compounded_requirements, sch, vol, output_filepath)
                sample_rows = extract_file_sample(output_filepath)
                
                # Step B: Spin up your verification judge to score the corrected dataset records again
                if process_judge:
                    configure_environment_keys(jd_mod, jd_url, jd_key)
                    fresh_audit_payload = pipeline.run_post_audit(jd_mod, original_reqs, sch, records)
                    return sample_rows, output_filepath, fresh_audit_payload, f"Refinement complete: Regenerated and re-audited dataset with critique feedback loop parameters."
                
                return sample_rows, output_filepath, {"status": "Audit skipped on refinement step."}, f"Refinement complete: Overwrote entries without executing a re-audit."
            except Exception as e:
                return [], None, {}, f"Refinement Error detected: {str(e)}"

        refine_btn.click(
            fn=refinement_loop_logic,
            inputs=[
                gen_model, gen_base_url, gen_api_key, 
                use_judge_cb, judge_model, judge_base_url, judge_api_key, 
                audit_view, config_input
            ],
            outputs=[output_preview, file_downloader, audit_view, status_view]
        )

if __name__ == "__main__":
    # Passing layout configuration assets with Gradio launch
    demo.launch(css=custom_css, theme=gr.themes.Soft())