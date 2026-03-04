# =========================
# IMPORTS
# =========================
import streamlit as st
import PyPDF2
from openai import OpenAI
import os
import time
import json
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

# =========================
# CONFIGURATION
# =========================
st.set_page_config(page_title="PDF Decision Extractor", layout="wide")
CHUNK_SIZE = 3000  # characters, approximate
OVERLAP_SIZE = 200  # characters to overlap between chunks
MODEL = "gpt-3.5-turbo"

# =========================
# PDF PROCESSING LAYER
# =========================
def extract_text_from_pdf(pdf_file):
    """Extract raw text from uploaded PDF safely."""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:  # Guard against None returns
                text += page_text
        return text
    except Exception as e:
        st.warning(f"PyPDF2 extraction failed: {str(e)}")
        return ""

def extract_text_with_ocr(pdf_file):
    """Fallback: convert PDF pages to images and extract text via OCR with progress."""
    text = ""
    try:
        pdf_file.seek(0)  # reset pointer
        # Use lower DPI to save memory on large PDFs
        images = convert_from_bytes(pdf_file.read(), dpi=200)
        
        total_pages = len(images)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, page in enumerate(images):
            status_text.text(f"OCR page {i+1}/{total_pages}...")
            page_text = pytesseract.image_to_string(page)
            if page_text:
                text += page_text + "\n"
            progress_bar.progress((i + 1) / total_pages)
        
        status_text.text("OCR complete!")
        return text
    except Exception as e:
        st.error(f"OCR failed: {str(e)}")
        return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP_SIZE):
    """Split text into paragraph-based chunks with overlap."""
    paragraphs = text.split("\n\n")  # simple paragraph split
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(para) + 2 > chunk_size:
            # Save current chunk if it exists
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous chunk
            if chunks:  # If there was a previous chunk
                overlap_text = chunks[-1][-overlap:] if len(chunks[-1]) > overlap else chunks[-1]
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para
        else:
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# =========================
# AI ENGINE LAYER
# =========================
def build_prompt(chunk):
    """Create structured JSON prompt for consistent extraction."""
    return f"""
    Extract all key decisions, action items, and important points from this text.
    
    Return ONLY valid JSON with this exact structure:
    {{
        "decisions": ["specific decision 1", "specific decision 2"],
        "action_items": ["action item 1", "action item 2"],
        "key_points": ["key point 1", "key point 2"]
    }}
    
    Rules:
    - If a category has no items, use empty list []
    - Be specific and concise
    - Extract directly from the text, don't invent
    - Return ONLY the JSON, no other text
    
    Text: {chunk}
    """

def call_ai(prompt, client):
    """Send prompt to model and return parsed JSON or fallback text."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        content = response.choices[0].message.content.strip()
        
        # Try to parse as JSON
        try:
            # Clean potential markdown code blocks
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            elif content.startswith('"""') and content.endswith('"""'):
                content = content.replace('"""', '').strip()
            
            return json.loads(content)
        except json.JSONDecodeError as e:
            # Fallback: return as raw text with structure note
            return {
                "raw_content": content, 
                "note": "AI didn't return valid JSON",
                "decisions": [],
                "action_items": [],
                "key_points": [f"Raw extraction (not structured): {content[:200]}..."]
            }
            
    except Exception as e:
        return {
            "error": str(e),
            "decisions": [],
            "action_items": [],
            "key_points": [f"Error processing chunk: {str(e)}"]
        }

def merge_results(results):
    """Combine multiple chunk results into one unified structure."""
    merged = {
        "decisions": [],
        "action_items": [],
        "key_points": []
    }
    
    for result in results:
        if isinstance(result, dict):
            # If it's our expected structure
            for key in merged.keys():
                if key in result and isinstance(result[key], list):
                    merged[key].extend(result[key])
            
            # Handle raw_content fallback
            if "raw_content" in result and result["raw_content"]:
                merged["key_points"].append(f"[Raw chunk]: {result['raw_content'][:100]}...")
    
    # Remove duplicates while preserving order
    for key in merged.keys():
        seen = set()
        unique = []
        for item in merged[key]:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        merged[key] = unique
    
    return merged

def process_document(chunks, client):
    """Process all chunks and merge results."""
    chunk_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"Processing chunk {i+1}/{len(chunks)}...")
        prompt = build_prompt(chunk)
        result = call_ai(prompt, client)
        chunk_results.append(result)
        progress_bar.progress((i + 1) / len(chunks))
    
    status_text.text("Merging results...")
    final_result = merge_results(chunk_results)
    return final_result

# =========================
# UI LAYER
# =========================
def render_output(result):
    """Display structured decision results cleanly."""
    st.markdown("## 📋 Extracted Decisions")
    
    # Handle error case
    if "error" in result and result["error"]:
        st.error(f"Error: {result['error']}")
        return
    
    # Display in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Decisions")
        if result.get("decisions") and len(result["decisions"]) > 0:
            for d in result["decisions"]:
                st.markdown(f"- {d}")
        else:
            st.markdown("*No decisions found*")
        
        st.markdown("### ⚡ Action Items")
        if result.get("action_items") and len(result["action_items"]) > 0:
            for a in result["action_items"]:
                st.markdown(f"- {a}")
        else:
            st.markdown("*No action items found*")
    
    with col2:
        st.markdown("### 💡 Key Points")
        if result.get("key_points") and len(result["key_points"]) > 0:
            for k in result["key_points"]:
                st.markdown(f"- {k}")
        else:
            st.markdown("*No key points found*")
    
    # Prepare text version for download
    text_output = ""
    for category, items in result.items():
        if category == "error":
            continue
        if isinstance(items, list) and len(items) > 0:
            text_output += f"\n{category.upper().replace('_', ' ')}\n"
            text_output += "-" * 30 + "\n"
            for item in items:
                text_output += f"• {item}\n"
            text_output += "\n"
    
    # Download button
    st.download_button(
        label="📥 Download as Text",
        data=text_output,
        file_name="extracted_decisions.txt",
        mime="text/plain"
    )

def get_api_key():
    """Retrieve API key from secrets or environment variables."""
    try:
        return st.secrets["OPENAI_API_KEY"]
    except:
        return os.getenv("OPENAI_API_KEY", "")

def main():
    """Main app entry point."""
    st.title("📄 PDF Decision Extractor")
    st.markdown("Upload a PDF to extract key decisions, actions, and insights.")
    
    # Get API key
    api_key = get_api_key()
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        if not api_key:
            api_key = st.text_input("OpenAI API Key", type="password")
        else:
            st.success("✅ API key loaded from secrets/env")
        
        st.markdown("---")
        st.markdown("**How it works:**")
        st.markdown("1. Upload PDF")
        st.markdown("2. Text is extracted and chunked")
        st.markdown("3. AI extracts structured data")
        st.markdown("4. Results are merged and displayed")
        
        st.markdown("---")
        st.markdown("**Tips:**")
        st.markdown("- Scanned PDFs use OCR (slower)")
        st.markdown("- Large files may take a few minutes")
        st.markdown("- Results are saved in session")
    
    # Main area
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file:
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
        else:
            if st.button("🚀 Extract Decisions", type="primary"):
                with st.spinner("Processing your PDF..."):
                    # Initialize client
                    client = OpenAI(api_key=api_key)
                    
                    # Extract text with progress
                    with st.status("Extracting text from PDF..."):
                        text = extract_text_from_pdf(uploaded_file)
                    
                    if not text or not text.strip():
                        st.warning("No text detected — trying OCR fallback...")
                        with st.status("Running OCR (this may take a while)..."):
                            text = extract_text_with_ocr(uploaded_file)
                    
                    if not text or not text.strip():
                        st.error("❌ Could not extract text from this PDF. Try a different file.")
                        st.stop()
                    
                    # Show stats for whichever extraction succeeded
                    st.info(f"📊 Extracted {len(text)} characters, {len(text.split())} words")
                    
                    # Chunk text
                    chunks = chunk_text(text)
                    st.info(f"📦 Split into {len(chunks)} chunks for processing")
                    
                    # Process with AI
                    result = process_document(chunks, client)
                    
                    # Display
                    render_output(result)
                    
                    # Success message
                    st.success("✅ Extraction complete!")
    
    else:
        st.info("👆 Start by uploading a PDF file")

if __name__ == "__main__":
    main()
