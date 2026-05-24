import gradio as gr

def build_interface(pipeline_callback_fn):
    with gr.Blocks(title="Synthetic Documents Generator") as demo:
        
        # 1. HEADER SECTION
        gr.Markdown("# Synthetic Documents Generator")
        gr.Markdown(
            "An production ready synthetic documents generator. "
            "A planner model builds an entire multi-document schema and blueprint, "
            "and another model uses a high-performance parallel thread pool to build final documents archive."
        )
        
        gr.HTML("<hr style='border: 1px solid #e5e7eb; margin: 20px 0;'>")
        
        # 2. INPUT SECTION
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Context & Files Spec")
                topic_input = gr.Textbox(
                    label="Documents System Context / Prompt",
                    placeholder="Paste your comprehensive document generation prompt or context here...",
                    lines=8,
                    max_lines=15
                )
                execute_btn = gr.Button("🚀 Generate Complete Ecosystem Zip Archive", variant="primary", size="lg")
        
        gr.HTML("<hr style='border: 1px solid #e5e7eb; margin: 20px 0;'>")
        
        # 3. PLANNER OUTPUT SECTION
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Planner Blueprint & Summary")
                spec_output_markdown = gr.Markdown(
                    value="*Awaiting execution. Planner's document generation blueprint and summary will appear below...*"
                )
        
        gr.HTML("<hr style='border: 1px solid #e5e7eb; margin: 20px 0;'>")
        
        # 4. LOWER COMPILATION LOG & DOWNLOAD SECTION
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Processing Stream Output Log")
                document_output_markdown = gr.Markdown(
                    value="*Worker thread execution log will be displayed below...*"
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Generated Documents Export")
                file_download_element = gr.File(
                    label="Download Archive...",
                    interactive=False
                )

        execute_btn.click(
            fn=pipeline_callback_fn,
            inputs=[topic_input],
            outputs=[spec_output_markdown, document_output_markdown, file_download_element]
        )
        
    return demo