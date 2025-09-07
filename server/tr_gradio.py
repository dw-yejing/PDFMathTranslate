#!/usr/bin/env python3
"""
Gradio interface for PDF translation using pdf2zh
"""

import asyncio
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional, Tuple
import logging

import gradio as gr
import requests
import tqdm
from gradio_pdf import PDF
from string import Template

from pdf2zh import __version__
from pdf2zh.high_level import translate, download_remote_fonts
from pdf2zh.doclayout import ModelInstance, OnnxModel
from pdf2zh.config import ConfigManager
from pdf2zh.translator import (
    AnythingLLMTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DifyTranslator,
    ArgosTranslator,
    GeminiTranslator,
    GoogleTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAITranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
    GrokTranslator,
    GroqTranslator,
    DeepseekTranslator,
    OpenAIlikedTranslator,
    QwenMtTranslator,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Translation service mapping
SERVICE_MAP = {
    "Google": GoogleTranslator,
    "Bing": BingTranslator,
    "DeepL": DeepLTranslator,
    "DeepLX": DeepLXTranslator,
    "Ollama": OllamaTranslator,
    "Xinference": XinferenceTranslator,
    "AzureOpenAI": AzureOpenAITranslator,
    "OpenAI": OpenAITranslator,
    "Zhipu": ZhipuTranslator,
    "ModelScope": ModelScopeTranslator,
    "Silicon": SiliconTranslator,
    "Gemini": GeminiTranslator,
    "Azure": AzureTranslator,
    "Tencent": TencentTranslator,
    "Dify": DifyTranslator,
    "AnythingLLM": AnythingLLMTranslator,
    "Argos Translate": ArgosTranslator,
    "Grok": GrokTranslator,
    "Groq": GroqTranslator,
    "DeepSeek": DeepseekTranslator,
    "OpenAI-liked": OpenAIlikedTranslator,
    "Ali Qwen-Translation": QwenMtTranslator,
}

# Language mapping
LANG_MAP = {
    "Simplified Chinese": "zh",
    "Traditional Chinese": "zh-TW",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Korean": "ko",
    "Russian": "ru",
    "Spanish": "es",
    "Italian": "it",
}

# Page range mapping
PAGE_MAP = {
    "All pages": None,
    "First page": [0],
    "First 5 pages": list(range(0, 5)),
    "Custom range": None,
}

def download_with_limit(url: str, save_dir: Path, size_limit: int = 100 * 1024 * 1024) -> str:
    """Download file with size limit"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        if total_size > size_limit:
            raise ValueError(f"File size ({total_size} bytes) exceeds limit ({size_limit} bytes)")
        
        # Generate filename from URL
        filename = f"downloaded_{uuid.uuid4()}.pdf"
        file_path = save_dir / filename
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return str(file_path)
    except Exception as e:
        raise ValueError(f"Failed to download file: {str(e)}")

def parse_page_range(page_input: str) -> Optional[List[int]]:
    """Parse page range input"""
    if not page_input.strip():
        return None
    
    pages = []
    for p in page_input.split(","):
        p = p.strip()
        if "-" in p:
            start, end = p.split("-")
            pages.extend(range(int(start) - 1, int(end)))
        else:
            pages.append(int(p) - 1)
    
    return pages

def get_translator_envs(service: str) -> List[gr.components.Component]:
    """Get environment variable inputs for specific translation service"""
    if service not in SERVICE_MAP:
        return []
    
    translator_class = SERVICE_MAP[service]
    envs = getattr(translator_class, 'envs', {})
    
    components = []
    for key, default_value in envs.items():
        if default_value is None:
            components.append(gr.Textbox(label=key, placeholder=f"Enter {key}", type="password"))
        else:
            components.append(gr.Textbox(label=key, value=str(default_value)))
    
    return gr.Accordion.update(
        open=True,
        children=components
    )

def translate_pdf(
    file_input,
    link_input,
    service: str,
    lang_from: str,
    lang_to: str,
    page_range: str,
    page_input: str,
    prompt: str,
    threads: int,
    skip_subset_fonts: bool,
    ignore_cache: bool,
    vfont: str,
    vchar: str,
    progress=gr.Progress(),
    *envs
) -> Tuple[str, str, str, str]:
    """
    Translate PDF file using pdf2zh
    
    Returns:
        Tuple of (mono_pdf_path, dual_pdf_path, log_output)
    """
    try:
        # Validate inputs
        if not file_input and not link_input:
            return None, None, "Error: Please provide either a file or a link", "Error: No input provided"
        
        # Setup output directory
        output = Path("pdf2zh_files")
        output.mkdir(parents=True, exist_ok=True)
        
        # Get file path
        if file_input:
            file_path = shutil.copy(file_input, output)
        else:
            # Download from link
            file_path = download_with_limit(link_input, output)
        
        filename = os.path.splitext(os.path.basename(file_path))[0]
        file_mono = output / f"{filename}-mono.pdf"
        file_dual = output / f"{filename}-dual.pdf"
        
        # Parse parameters
        lang_in = LANG_MAP.get(lang_from, lang_from)
        lang_out = LANG_MAP.get(lang_to, lang_to)
        
        # Parse page range
        if page_range == "Custom range":
            pages = parse_page_range(page_input)
        else:
            pages = PAGE_MAP.get(page_range)
        
        # Setup environment variables
        env_dict = {}
        if service in SERVICE_MAP:
            translator_class = SERVICE_MAP[service]
            env_keys = list(getattr(translator_class, 'envs', {}).keys())
            for i, key in enumerate(env_keys):
                if i < len(envs) and envs[i]:
                    env_dict[key] = envs[i]
        
        # Setup prompt template
        prompt_template = None
        if prompt:
            prompt_template = Template(prompt)
        
        # Load model
        model = ModelInstance.value or OnnxModel.load_available()
        
        # Setup progress callback
        def progress_callback(progress_obj):
            if hasattr(progress_obj, 'n'):
                progress(progress_obj.n / progress_obj.total)
        
        # Perform translation
        progress(0, desc="Starting translation...")
        
        result_files = translate(
            files=[file_path],
            output=str(output),
            pages=pages,
            lang_in=lang_in,
            lang_out=lang_out,
            service=service.lower(),
            thread=threads,
            vfont=vfont,
            vchar=vchar,
            callback=progress_callback,
            model=model,
            envs=env_dict,
            prompt=prompt_template,
            skip_subset_fonts=skip_subset_fonts,
            ignore_cache=ignore_cache,
        )
        
        progress(1.0, desc="Translation completed!")
        
        if result_files:
            mono_path, dual_path = result_files[0]
            
            
            return mono_path, dual_path, "Translation completed successfully!", "Translation completed successfully!"
        else:
            return None, None, "Translation failed - no output files generated", "Translation failed"
            
    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        return None, None, f"Error during translation: {str(e)}", f"Error: {str(e)}"

def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="PDF Translation Tool", theme=gr.themes.Soft()) as interface:
        gr.Markdown(f"# PDF Translation Tool v{__version__}")
        gr.Markdown("Translate PDF documents using various translation services")
        
        with gr.Row():
            with gr.Column(scale=1):
                # File input section
                gr.Markdown("## Input")
                
                with gr.Tab("Upload File"):
                    file_input = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                
                with gr.Tab("Download from URL"):
                    link_input = gr.Textbox(
                        label="PDF URL",
                        placeholder="https://example.com/document.pdf"
                    )
                
                # Translation settings
                gr.Markdown("## Translation Settings")
                
                with gr.Row():
                    service = gr.Dropdown(
                        choices=list(SERVICE_MAP.keys()),
                        value="Google",
                        label="Translation Service"
                    )
                envs = []
                for i in range(3):
                    envs.append(
                        gr.Textbox(
                            visible=False,
                            interactive=True,
                        )
                    )

                with gr.Row():
                    lang_from = gr.Dropdown(
                        choices=list(LANG_MAP.keys()),
                        value="English",
                        label="Source Language"
                    )
                    lang_to = gr.Dropdown(
                        choices=list(LANG_MAP.keys()),
                        value="Simplified Chinese",
                        label="Target Language"
                    )
                
                with gr.Row():
                    page_range = gr.Dropdown(
                        choices=list(PAGE_MAP.keys()),
                        value="All pages",
                        label="Page Range"
                    )
                    page_input = gr.Textbox(
                        label="Custom Page Range (e.g., 1-5, 10, 15-20)",
                        placeholder="Leave empty for all pages",
                        visible=False
                    )
                
                # Advanced settings
                with gr.Accordion("Advanced Settings", open=False):
                    prompt = gr.Textbox(
                        label="Custom Prompt",
                        placeholder="Enter custom translation prompt...",
                        lines=3
                    )
                    
                    with gr.Row():
                        threads = gr.Slider(
                            minimum=1,
                            maximum=16,
                            value=4,
                            step=1,
                            label="Threads"
                        )
                        skip_subset_fonts = gr.Checkbox(
                            label="Skip Font Subsetting",
                            value=False
                        )
                    
                    with gr.Row():
                        ignore_cache = gr.Checkbox(
                            label="Ignore Cache",
                            value=False
                        )
                        vfont = gr.Textbox(
                            label="Formula Font Regex",
                            placeholder="Enter regex for formula fonts"
                        )
                    
                    vchar = gr.Textbox(
                        label="Formula Character Regex",
                        placeholder="Enter regex for formula characters"
                    )
                
                # Service-specific environment variables
                env_components = gr.Accordion("Service Configuration", open=False)
                
                # Update environment variables when service changes
                def update_env_components(service_name): #yj
                    components = get_translator_envs(service_name)
                    return components
                
                service.change(
                    update_env_components,
                    service,
                    envs
                )
                
                # Update page input visibility
                def update_page_input_visibility(choice):
                    return gr.Textbox(visible=choice == "Custom range")
                
                page_range.change(
                    update_page_input_visibility,
                    inputs=[page_range],
                    outputs=[page_input]
                )
                
                # Translate button
                translate_btn = gr.Button("Translate PDF", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("## Output")
                
                status_output = gr.Textbox(
                    label="Status",
                    value="Ready to translate",
                    interactive=False
                )
                
                log_output = gr.Textbox(
                    label="Translation Log",
                    lines=8,
                    interactive=False
                )
                
                with gr.Row():
                    mono_pdf = gr.File(
                        label="Mono PDF (Original + Translation)",
                        file_types=[".pdf"],
                        interactive=False
                    )
                    dual_pdf = gr.File(
                        label="Dual PDF (Side by Side)",
                        file_types=[".pdf"],
                        interactive=False
                    )
        
        # Connect translate button
        translate_btn.click(
            translate_pdf,
            inputs=[
                file_input, link_input, service, lang_from, lang_to,
                page_range, page_input, prompt, threads, skip_subset_fonts,
                ignore_cache, vfont, vchar, *env_components.children
            ],
            outputs=[mono_pdf, dual_pdf, log_output, status_output]
        )
        
        # Initial environment components
        initial_envs = get_translator_envs("Google")
        env_components.children = initial_envs
    
    return interface

def main():
    """Main function to launch the Gradio interface"""
    # Initialize model
    ModelInstance.value = OnnxModel.load_available()
    
    # Create and launch interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    envs = []
    main()
