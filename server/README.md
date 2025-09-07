# PDF Translation Gradio Interface

This directory contains a Gradio web interface for the PDF translation functionality using pdf2zh.

## Features

- **File Upload**: Upload PDF files directly through the web interface
- **URL Download**: Download PDFs from URLs
- **Multiple Translation Services**: Support for various translation services including:
  - Google Translate
  - Bing Translator
  - DeepL
  - OpenAI
  - Azure OpenAI
  - And many more...
- **Language Support**: Multiple source and target languages
- **Page Range Selection**: Translate specific pages or entire documents
- **Advanced Settings**: Custom prompts, thread control, font settings
- **Progress Tracking**: Real-time progress updates
- **Dual Output**: Generate both mono (original + translation) and dual (side-by-side) PDFs

## Usage

### Running the Interface

```bash
# From the project root
python server/tr_gradio.py

# Or with custom port
python server/tr_gradio.py --port 8080
```

The interface will be available at `http://localhost:7860` by default.

### Interface Components

1. **Input Section**:
   - Upload PDF file or provide URL
   - Choose translation service
   - Select source and target languages
   - Specify page range

2. **Advanced Settings**:
   - Custom translation prompts
   - Thread count for parallel processing
   - Font subsetting options
   - Cache control
   - Formula font/character regex patterns

3. **Service Configuration**:
   - API keys and endpoints for various services
   - Service-specific settings

4. **Output Section**:
   - Status updates
   - Translation log
   - Download links for translated PDFs

### Translation Services

The interface supports multiple translation services:

- **Google**: Free, no API key required
- **Bing**: Free, no API key required
- **DeepL**: Requires API key
- **OpenAI**: Requires API key
- **Azure OpenAI**: Requires Azure OpenAI API key
- **Ollama**: Local translation using Ollama models
- **Xinference**: Local translation using Xinference models
- And many more...

### Configuration

For services requiring API keys, you can:

1. Set environment variables
2. Use the web interface to input API keys
3. Configure in the pdf2zh config file

### Output Files

The translation process generates two types of output:

1. **Mono PDF**: Original text with translations inline
2. **Dual PDF**: Side-by-side comparison of original and translated text

Files are saved in the `pdf2zh_files` directory.

## Dependencies

The interface requires the following dependencies (already included in the main project):

- `gradio<5.36`
- `gradio_pdf>=0.0.21`
- `requests`
- All pdf2zh dependencies

## Troubleshooting

### Common Issues

1. **Model Loading**: Ensure the ONNX model is available
2. **API Keys**: Verify API keys for paid services
3. **File Size**: Large PDFs may take longer to process
4. **Memory**: Ensure sufficient RAM for large documents

### Error Messages

- "No input provided": Upload a file or provide a URL
- "Translation failed": Check service configuration and API keys
- "File size exceeds limit": Use smaller files or increase limits

## Development

The interface is built using Gradio and integrates with the pdf2zh library. Key components:

- `translate_pdf()`: Main translation function
- `create_interface()`: Gradio interface setup
- Service-specific environment handling
- Progress tracking and error handling

## License

Same as the main pdf2zh project (AGPL-3.0). 