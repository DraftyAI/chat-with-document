import streamlit as st
from dotenv import load_dotenv
import base64
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# Import FAISS with error handling
try:
    from langchain_community.vectorstores import FAISS
    FAISS_AVAILABLE = True
except ImportError:
    print("Error importing FAISS. Please install it with 'pip install faiss-cpu' or 'pip install faiss-gpu'")
    FAISS_AVAILABLE = False
    # Import a fallback vectorstore
    from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document, BaseRetriever
from langchain_community.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template
import boto3
import io
import json
import time
import re
import os
import uuid
from typing import List

# Load environment variables
load_dotenv()

# Constants
ENVIRONMENT = os.getenv('ENVIRONMENT', 'dev')
S3_BUCKET_NAME = f'draftyai-textract-chat-with-docs-{ENVIRONMENT}'

# Set page configuration
st.set_page_config(
    page_title="Chat with FOIA - DraftyAI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants for styling
BACKGROUND_COLOR = 'white'
COLOR = 'black'

def set_page_container_style(
        has_document: bool = False,
        max_width: int = 1100, 
        max_width_100_percent: bool = True,
        padding_top: int = 1, 
        padding_right: int = 1, 
        padding_left: int = 1, 
        padding_bottom: int = 1,
        color: str = COLOR, 
        background_color: str = BACKGROUND_COLOR,
    ):
        if max_width_100_percent:
            max_width_str = f'max-width: 100%;'
        else:
            max_width_str = f'max-width: {max_width}px;'
        
        # Different padding for document/no-document states
        doc_padding = "1rem" if has_document else "6rem"
        
        st.markdown(
            f'''
            <style>
                .reportview-container .css-1lcbmhc .css-1outpf7 {{
                    padding-top: 35px;
                }}
                .reportview-container .main .block-container {{
                    {max_width_str}
                    padding-top: {padding_top}rem;
                    padding-right: {padding_right}rem;
                    padding-left: {padding_left}rem;
                    padding-bottom: {padding_bottom}rem;
                }}
                .reportview-container .main {{
                    color: {color};
                    background-color: {background_color};
                }}
                .block-container {{
                    padding-top: {doc_padding} !important;
                    padding-bottom: 0rem !important;
                }}
                [data-testid="stSidebar"] {{
                    padding-top: 0rem !important;
                }}
            </style>
            ''',
            unsafe_allow_html=True,
        )

# Apply the custom styling
def apply_custom_styling(has_document):
    set_page_container_style(has_document=has_document)

def get_boto3_client(service_name):
    """Create a boto3 client with profile or credentials fallback and region."""
    region_name = os.getenv('AWS_REGION', 'us-east-1')
    profile_name = os.getenv('AWS_PROFILE')
    
    print(f"Using AWS region: {region_name}")
    
    if profile_name:
        print(f"Using AWS profile: {profile_name}")
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        return session.client(service_name)
    else:
        print("Using AWS access keys from environment variables")
        # This will automatically use AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables
        return boto3.client(service_name, region_name=region_name)

async def extract_text_with_textract(pdf_file, file_name):
    # Initialize AWS clients
    textract = get_boto3_client('textract')
    s3 = get_boto3_client('s3')
    
    # Read the PDF file into bytes
    pdf_bytes = pdf_file.read()
    st.write(f"PDF size: {len(pdf_bytes)} bytes")
    
    try:
        # Generate a unique filename to avoid collisions
        unique_filename = f"{str(uuid.uuid4())}-{file_name}"
        st.write(f"Uploading to S3 as: {unique_filename}")
        
        # Upload to S3
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=unique_filename,
            Body=pdf_bytes
        )
        
        # Start async document text detection
        st.write("Starting async document text detection with AWS Textract...")
        response = textract.start_document_text_detection(
            DocumentLocation={
                'S3Object': {
                    'Bucket': S3_BUCKET_NAME,
                    'Name': unique_filename
                }
            }
        )
        
        job_id = response['JobId']
        st.write(f"Started Textract job: {job_id}")
        
        # Wait for the job to complete
        while True:
            response = textract.get_document_text_detection(JobId=job_id)
            status = response['JobStatus']
            st.write(f"Job status: {status}")
            
            if status in ['SUCCEEDED', 'FAILED']:
                break
                
            time.sleep(3)
        
        try:
            # Clean up the S3 file regardless of success/failure
            s3.delete_object(
                Bucket=S3_BUCKET_NAME,
                Key=unique_filename
            )
        except Exception as e:
            st.warning(f"Warning: Could not delete temporary S3 file: {str(e)}")
        
        if status == 'SUCCEEDED':
            # Get all blocks from all pages
            blocks = []
            next_token = None
            
            while True:
                if next_token:
                    response = textract.get_document_text_detection(JobId=job_id, NextToken=next_token)
                else:
                    response = textract.get_document_text_detection(JobId=job_id)
                
                blocks.extend(response['Blocks'])
                
                if 'NextToken' in response:
                    next_token = response['NextToken']
                else:
                    break
            
            # Process blocks to reconstruct text with page numbers
            text = ""
            chunk_locations = []
            current_page = None
            page_text = ""
            
            for block in blocks:
                if block['BlockType'] == 'PAGE':
                    # When we encounter a new page, save the previous page's text
                    if current_page is not None and page_text:
                        chunk_locations.append({
                            'file': file_name,
                            'page': current_page,
                            'text': page_text,
                            'start_char': len(text),
                            'end_char': len(text) + len(page_text)
                        })
                        text += page_text
                        page_text = ""
                    
                    current_page = block['Page']
                
                elif block['BlockType'] == 'LINE' and current_page is not None:
                    page_text += block['Text'] + "\n"
            
            # Don't forget to add the last page
            if current_page is not None and page_text:
                chunk_locations.append({
                    'file': file_name,
                    'page': current_page,
                    'text': page_text,
                    'start_char': len(text),
                    'end_char': len(text) + len(page_text)
                })
                text += page_text
            
            total_chars = len(text)
            st.write(f"Successfully extracted {total_chars} characters")
            
            return text, chunk_locations
            
        else:
            error_message = response.get('StatusMessage', 'Unknown error')
            raise Exception(f"Textract job failed: {error_message}")
            
    except Exception as e:
        st.error(f"Error processing document with Textract: {str(e)}")
        return ""

async def process_documents(pdf_docs):
    text = ""
    chunk_locations = []
    has_text = False
    
    for pdf in pdf_docs:
        # First try standard PDF text extraction
        pdf_reader = PdfReader(pdf)
        
        # Try to extract text from each page
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            
            # Print the number of characters found
            st.write(f"Page {page_num + 1}: {len(page_text)} characters")
            
            if page_text:
                has_text = True
                chunk_locations.append({
                    'file': pdf.name,
                    'page': page_num + 1,
                    'text': page_text,
                    'start_char': len(text),
                    'end_char': len(text) + len(page_text)
                })
                text += page_text
        
        if not has_text:
            st.write(f"No text found in '{pdf.name}' using standard extraction. Trying AWS Textract...")
            
            # Get PDF file size
            pdf.seek(0, 2)  # Seek to end
            pdf_size = pdf.tell()
            st.write(f"PDF size: {pdf_size} bytes")
            
            # Reset file pointer
            pdf.seek(0)
            
            # Try extracting text with Textract
            textract_text, textract_chunk_locations = await extract_text_with_textract(pdf, pdf.name)
            
            if textract_text:
                has_text = True
                chunk_locations.extend(textract_chunk_locations)
                text += textract_text
                st.success(f"Successfully extracted text from '{pdf.name}' using AWS Textract")
            else:
                st.error(f"Could not extract any text from '{pdf.name}' using either method")
    
    if not has_text:
        st.error("No text could be extracted from any of the uploaded documents")
        return None, None
    
    return text, chunk_locations

def get_text_chunks(text, chunk_locations):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_text(text)
    
    # Map chunks back to their source pages
    chunk_metadata = []
    for i, chunk in enumerate(chunks):
        chunk_start = text.find(chunk)
        chunk_end = chunk_start + len(chunk)
        
        # Find which original chunk(s) this piece came from
        relevant_pages = set()
        for loc in chunk_locations:
            if (chunk_start < loc['end_char'] and chunk_end > loc['start_char']):
                relevant_pages.add(loc['page'])
        
        chunk_metadata.append({
            'chunk_index': i,
            'pages': list(relevant_pages)
        })
    
    return chunks, chunk_metadata

def get_vectorstore(text_chunks, chunk_metadata):
    print("Starting vectorstore creation...")
    try:
        # First, check if FAISS is properly imported
        import sys
        print(f"Python version: {sys.version}")
        
        # Try to import faiss directly to check if it's accessible
        try:
            import faiss
            print(f"FAISS version: {getattr(faiss, '__version__', 'unknown')}")
        except ImportError as e:
            print(f"Direct FAISS import failed: {str(e)}")
        except Exception as e:
            print(f"FAISS import error: {str(e)}")
        
        # Create embeddings
        print("Creating OpenAI embeddings...")
        embeddings = OpenAIEmbeddings()
        print("Embeddings created successfully")
        
        # Store chunk locations in metadata
        print(f"Creating {len(text_chunks)} document objects...")
        documents = [
            Document(
                page_content=chunk,
                metadata={'chunk_index': metadata['chunk_index'], 'pages': metadata['pages']}
            ) for chunk, metadata in zip(text_chunks, chunk_metadata)
        ]
        print("Document objects created successfully")
        
        # Create FAISS index
        if FAISS_AVAILABLE:
            print("Attempting to create FAISS index...")
            try:
                # Test basic FAISS functionality
                import faiss
                import numpy as np
                print("Testing basic FAISS functionality...")
                dimension = 128
                index = faiss.IndexFlatL2(dimension)
                print("Basic FAISS test successful")
                
                # Now try to create the actual vectorstore
                vectorstore = FAISS.from_documents(documents, embeddings)
                print("FAISS index created successfully")
            except Exception as e:
                print(f"FAISS index creation failed: {str(e)}")
                print("Falling back to Chroma...")
                vectorstore = Chroma.from_documents(documents, embeddings)
                print("Chroma index created successfully")
        else:
            print("FAISS not available, using Chroma as fallback...")
            vectorstore = Chroma.from_documents(documents, embeddings)
            print("Chroma index created successfully")
        return vectorstore
    except Exception as e:
        import traceback
        print(f"Error in get_vectorstore: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return None

class ScoredVectorStoreRetriever(BaseRetriever):
    def __init__(self, vectorstore):
        super().__init__()
        self._vectorstore = vectorstore
        
    def _get_relevant_documents(self, query: str) -> List:
        docs_and_scores = self._vectorstore.similarity_search_with_score(query, k=4)
        docs = []
        for doc, score in docs_and_scores:
            doc.metadata['score'] = score
            docs.append(doc)
        return docs
        
    async def _aget_relevant_documents(self, query: str) -> List:
        raise NotImplementedError("Async retrieval not implemented")

def get_conversation_chain(vectorstore):
    try:
        llm = ChatOpenAI()
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            output_key='answer'
        )
        
        # Verify vectorstore is valid
        if vectorstore is None:
            raise ValueError("Vector store is None, cannot create conversation chain")
        
        # Create our custom retriever that includes similarity scores
        retriever = ScoredVectorStoreRetriever(vectorstore)
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            output_key='answer'
        )
        
        return conversation_chain
    except Exception as e:
        print(f"Error creating conversation chain: {str(e)}")
        raise Exception(f"Failed to initialize conversation chain: {str(e)}")

def display_pdf(pdf_bytes, highlight_text=None, page_num=1, key=None):
    try:
        # Add logging for highlight_text
        print(f"display_pdf called with highlight_text: {highlight_text}, key: {key}")
        
        # Try to read with PyPDF2 to validate PDF
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            num_pages = len(reader.pages)
        except Exception as e:
            st.error(f"PDF validation error: {str(e)}")
            return

        # Encode PDF bytes to base64
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        
        # Add more logging for highlight_text
        if highlight_text:
            print(f"Highlight text before passing to JavaScript: '{highlight_text}'")
            # Escape any special characters for JavaScript
            highlight_text = highlight_text.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')
            print(f"Escaped highlight text: '{highlight_text}'")
        else:
            print("No highlight text to pass to JavaScript")
            # Set to empty string instead of None for JavaScript
            highlight_text = ""
        
        # Create viewer HTML
        pdf_display = f'''
            <div style="position: sticky; top: 0; background: white; z-index: 10; padding: 1px; border-bottom: 1px solid #ddd;">
                <div id="status" style="display: none; position: fixed; top: 20px; left: 50%; transform: translateX(-50%); 
                    padding: 8px 16px; background: rgba(0, 0, 0, 0.8); color: white; border-radius: 4px; 
                    transition: opacity 0.3s ease-in-out; z-index: 1000;">Loading viewer...</div>
                <div id="error" style="display: none; margin-bottom: 10px; padding: 10px; background: #ffe6e6; border-radius: 4px; color: red;"></div>
                <div id="zoom-controls" style="margin-bottom: 10px;">
                    <button onclick="zoomIn()" style="margin-right: 10px; padding: 5px 10px;">-</button>
                    <span id="zoom-level" style="margin-right: 10px;">100%</span>
                    <button onclick="zoomOut()" style="margin-right: 10px; padding: 5px 10px;">+</button>
                    <button onclick="resetZoom()" style="padding: 5px 10px;">Autofit</button>
                </div>
            </div>
            <div id="pdf-container" style="position: relative;"></div>

            <script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js"></script>
            <script>
                (function() {{ // Wrap in IIFE to avoid global scope pollution
                    pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
                    
                    let currentScale = 1.0;
                    let defaultScale = 1.0;
                    let pdfDocument = null;
                    
                    function updateZoomLevel() {{
                        document.getElementById('zoom-level').textContent = Math.round(currentScale * 100) + '%';
                    }}
                    
                    function zoomIn() {{
                        currentScale *= 1.2;
                        renderAllPages();
                    }}
                    
                    function zoomOut() {{
                        currentScale *= 0.8;
                        renderAllPages();
                    }}
                    
                    function resetZoom() {{
                        currentScale = defaultScale;
                        renderAllPages();
                    }}
                    
                    function calculateFitScale(page, container) {{
                        const viewport = page.getViewport({{ scale: 1.0 }});
                        const containerWidth = container.clientWidth;
                        const containerHeight = container.clientHeight;
                        
                        // Calculate scale to fit width with some padding
                        const scaleWidth = (containerWidth - 40) / viewport.width;
                        
                        // Use width-based scale as default
                        return scaleWidth;
                    }}

                    function scrollToPage(pageNum) {{
                        console.log('scrollToPage called with pageNum:', pageNum);
                        const container = document.getElementById('pdf-container');
                        const pages = container.children;
                        console.log('Found', pages.length, 'pages in container');
                        if (pageNum > 0 && pageNum <= pages.length) {{
                            const targetPage = pages[pageNum - 1];
                            if (targetPage) {{
                                console.log('Scrolling to page', pageNum);
                                targetPage.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                            }}
                        }} else {{
                            console.warn('Invalid page number:', pageNum, 'Total pages:', pages.length);
                        }}
                    }}

                    addEventListener('message', function(e) {{
                        console.log('PDF viewer received message:', e.data);
                        if (e.data.type === 'scrollToPage') {{
                            scrollToPage(e.data.pageNum);
                        }}
                    }}, false);

                    // Make zoom functions globally available
                    window.zoomIn = zoomIn;
                    window.zoomOut = zoomOut;
                    window.resetZoom = resetZoom;

                    async function renderPage(pdf, pageNum, container, highlight = false) {{
                        const page = await pdf.getPage(pageNum);
                        const viewport = page.getViewport({{ scale: currentScale }});

                        const pageDiv = document.createElement('div');
                        pageDiv.className = 'pdf-page';
                        pageDiv.setAttribute('data-page-number', pageNum);
                        pageDiv.style.margin = '20px auto';
                        pageDiv.style.position = 'relative';

                        const canvas = document.createElement('canvas');
                        canvas.style.display = 'block';
                        canvas.style.margin = '0 auto';
                        canvas.style.border = '1px solid #ddd';
                        canvas.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
                        
                        const context = canvas.getContext('2d');
                        canvas.height = viewport.height;
                        canvas.width = viewport.width;

                        pageDiv.appendChild(canvas);
                        container.appendChild(pageDiv);

                        await page.render({{
                            canvasContext: context,
                            viewport: viewport
                        }}).promise;

                        // If we have highlight text, try to highlight it
                        // We highlight if either the global highlight text is set or if the highlight parameter is true
                        if ((window.highlightText && window.highlightText.trim() !== '') || highlight) {{
                            console.log('Page', pageNum, 'checking highlight text:', window.highlightText);
                            console.log('Is highlight text empty?', !window.highlightText || window.highlightText.trim() === '');
                            console.log('Highlight text type:', typeof window.highlightText);
                            console.log('Highlight text length:', window.highlightText ? window.highlightText.length : 0);
                            
                            // Check that highlight text is not empty and not the string "None"
                            if (window.highlightText && window.highlightText.trim() !== '' && window.highlightText !== 'None') {{
                                console.log('Highlighting text on page', pageNum, ':', window.highlightText);
                                const textContent = await page.getTextContent();
                                const textItems = textContent.items;
                                
                                console.log('Found', textItems.length, 'text items on page', pageNum);
                                
                                // Keep track of whether we found any matches
                                let foundMatch = false;
                                let matchCount = 0;
                                
                                // Break the highlight text into words for more flexible matching
                                const highlightWords = window.highlightText.toLowerCase().split(/\s+/).filter(word => word.length > 3);
                                console.log('Searching for words:', highlightWords);
                                
                                // Highlight any text item that contains any of the significant words
                                textItems.forEach(function(textItem, index) {{
                                    // Log every 20th item to avoid console spam
                                    if (index % 20 === 0) {{
                                        console.log('Sample text item', index, ':', textItem.str);
                                    }}
                                    
                                    const itemText = textItem.str.toLowerCase();
                                    
                                    // Check if this text item contains any of our significant words
                                    let shouldHighlight = false;
                                    for (const word of highlightWords) {{
                                        if (word.length > 3 && itemText.includes(word)) {{
                                            shouldHighlight = true;
                                            console.log('Found word match:', word, 'in:', itemText);
                                            break;
                                        }}
                                    }}
                                    
                                    // Direct match check as fallback
                                    if (!shouldHighlight && itemText.includes(window.highlightText.toLowerCase())) {{
                                        shouldHighlight = true;
                                        console.log('Found direct match in:', itemText);
                                    }}
                                    
                                    if (shouldHighlight) {{
                                        foundMatch = true;
                                        matchCount++;
                                        
                                        const tx = pdfjsLib.Util.transform(
                                            viewport.transform,
                                            textItem.transform
                                        );
                                        
                                        context.beginPath();
                                        context.rect(tx[4], tx[5], textItem.width * currentScale, textItem.height * currentScale);
                                        context.fillStyle = 'yellow';
                                        context.globalAlpha = 0.3;
                                        context.fill();
                                        context.globalAlpha = 1.0;
                                    }}
                                }});
                                
                                if (foundMatch) {{
                                    console.log('Found and highlighted', matchCount, 'matches on page', pageNum);
                                }} else {{
                                    console.log('No matches found on page', pageNum);
                                }}
                            }}
                        }}
                    }}

                    async function renderAllPages() {{
                        const container = document.getElementById('pdf-container');
                        container.innerHTML = '';
                        
                        for (let pageNum = 1; pageNum <= pdfDocument.numPages; pageNum++) {{
                            await renderPage(pdfDocument, pageNum, container, true);
                        }}
                        updateZoomLevel();
                    }}

                    async function loadPDF() {{
                        try {{
                            const pdfData = atob('{base64_pdf}');
                            const loadingTask = pdfjsLib.getDocument({{data: pdfData}});
                            
                            pdfDocument = await loadingTask.promise;
                            const status = document.getElementById('status');
                            status.style.display = 'block';
                            
                            // Check if we're highlighting (when clicking a source button)
                            if (window.highlightText && window.highlightText.trim() !== '') {{
                                status.textContent = 'Highlighting the Relevant Sources...';
                            }} else {{
                                status.textContent = 'Rendering PDF...';
                            }}
                            
                            // Calculate initial scale based on first page
                            const firstPage = await pdfDocument.getPage(1);
                            const container = document.getElementById('pdf-container');
                            defaultScale = calculateFitScale(firstPage, container);
                            currentScale = defaultScale;
                            
                            await renderAllPages();
                            
                            // If we're highlighting, don't show the success message
                            if (!(window.highlightText && window.highlightText.trim() !== '')) {{
                                status.textContent = `PDF loaded successfully (${{pdfDocument.numPages}} pages)`;
                                
                                // Hide status after 5 seconds
                                setTimeout(() => {{
                                    status.style.opacity = '0';
                                    setTimeout(() => {{
                                        status.style.display = 'none';
                                        status.style.opacity = '1';
                                    }}, 300); // Wait for fade out animation
                                }}, 5000);
                            }} else {{
                                // For highlighting, hide status immediately after rendering
                                setTimeout(() => {{
                                    status.style.opacity = '0';
                                    setTimeout(() => {{
                                        status.style.display = 'none';
                                        status.style.opacity = '1';
                                    }}, 300); // Wait for fade out animation
                                }}, 1000); // Just 1 second for highlighting
                            }}
                            
                            scrollToPage({page_num});
                            
                        }} catch (error) {{
                            document.getElementById('error').style.display = 'block';
                            document.getElementById('error').textContent = 'Error: ' + error.message;
                            console.error('Error:', error);
                        }}
                    }}

                    // Listen for highlight update events from source buttons
                    document.addEventListener('updatePdfHighlight', function(e) {{
                        if (pdfDocument) {{
                            const highlightText = e.detail.highlightText;
                            const targetPage = e.detail.targetPage;
                            
                            console.log('Received updatePdfHighlight event:', highlightText, targetPage);
                            
                            // Update the highlight text
                            window.highlightText = highlightText;
                            
                            // Scroll to the target page
                            scrollToPage(targetPage);
                            
                            // Re-render the page with the new highlight
                            renderPage(pdfDocument, targetPage, document.getElementById('pdf-container'), true);
                            
                            // Clear the stored values
                            sessionStorage.removeItem('highlightText');
                            sessionStorage.removeItem('targetPage');
                        }}
                    }});
                    
                    // Add resize handler
                    let resizeTimeout;
                    addEventListener('resize', function() {{
                        // Debounce the resize event
                        clearTimeout(resizeTimeout);
                        resizeTimeout = setTimeout(async function() {{
                            if (pdfDocument) {{
                                const firstPage = await pdfDocument.getPage(1);
                                const container = document.getElementById('pdf-container');
                                if (currentScale === defaultScale) {{
                                    // Only auto-adjust if we're at the default (autofit) scale
                                    defaultScale = calculateFitScale(firstPage, container);
                                    currentScale = defaultScale;
                                    renderAllPages();
                                }}
                            }}
                        }}, 250); // Wait 250ms after resize ends before recalculating
                    }});
                    
                    // Check for stored highlight on page load
                    window.addEventListener('load', function() {{
                        const storedHighlight = sessionStorage.getItem('highlightText');
                        const storedPage = sessionStorage.getItem('targetPage');
                        
                        if (storedHighlight && storedPage && pdfDocument) {{
                            console.log('Found stored highlight:', storedHighlight, storedPage);
                            
                            // Update the highlight text
                            window.highlightText = storedHighlight;
                            
                            // Scroll to the target page
                            scrollToPage(parseInt(storedPage));
                            
                            // Re-render the page with the new highlight
                            renderPage(pdfDocument, parseInt(storedPage), document.getElementById('pdf-container'), true);
                            
                            // Clear the stored values
                            sessionStorage.removeItem('highlightText');
                            sessionStorage.removeItem('targetPage');
                        }}
                    }});
                    
                    // Initialize with the highlight text
                    window.highlightText = '{highlight_text}';
                    
                    // Make highlight text accessible globally
                    window.updateHighlightText = function(text, page) {{
                        window.highlightText = text;
                        if (pdfDocument) {{
                            scrollToPage(page);
                            renderPage(pdfDocument, page, document.getElementById('pdf-container'), true);
                        }}
                    }};
                    
                    loadPDF();
                }})(); // End IIFE
            </script>
        '''
        
        st.components.v1.html(pdf_display, height=800, scrolling=True)
        
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

def handle_userinput(user_question):
    if not user_question:
        return

    # Check if this is a new question
    if 'current_question' not in st.session_state or st.session_state.current_question != user_question:
        if st.session_state.conversation is None:
            st.error("Please process the documents first!")
            return
        
        # Clear any existing highlight when asking a new question
        st.session_state.highlight_text = ""  # Empty string instead of None
            
        st.session_state.is_loading = True
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        st.session_state.current_question = user_question
        if 'source_documents' in response:
            st.session_state.current_source_docs = response['source_documents']
            # Log source documents and their metadata
            print(f"Found {len(response['source_documents'])} source documents")
            for i, doc in enumerate(response['source_documents']):
                print(f"Source {i+1} metadata: {doc.metadata}")
                chunk_index = doc.metadata.get('chunk_index')
                if chunk_index is not None and chunk_index < len(st.session_state.chunk_locations):
                    print(f"  Chunk location found: {st.session_state.chunk_locations[chunk_index]}")
                else:
                    print(f"  No chunk location found for index {chunk_index}")
        else:
            st.session_state.current_source_docs = []
            print("No source documents found in response")
        
        st.session_state.is_loading = False
        # Force a rerun to update the chat history
        st.rerun()

def display_chat_history():
    if not st.session_state.chat_history:
        return
    
    # Create a scrolling container for chat messages with fixed height
    chat_container = st.container(height=650)
    with chat_container:
        # Keep track of which conversation pair we're on
        conversation_idx = 0
        for i, message in enumerate(st.session_state.chat_history):
            # Update conversation index for each user message (even indices)
            if i % 2 == 0:
                conversation_idx = i // 2
            
            if i % 2 == 0:
                st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                bot_message = bot_template.replace("{{MSG}}", message.content)
                st.markdown(bot_message, unsafe_allow_html=True)
                
                # Add sources explanation and display sources
                if st.session_state.current_source_docs:
                    st.markdown("""
                        <div style="margin-top: 10px; margin-bottom: 10px; padding: 10px; border-radius: 5px; background-color: #f0f2f6;">
                            <p style="margin: 0; font-size: 0.9em;">
                                <strong>📚 Sources:</strong> Listed below in order of relevance to your question. 
                                The relevance score indicates how closely each source matches your question.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display source references with relevance scores
                    for idx, doc in enumerate(st.session_state.current_source_docs, 1):
                        # Get pages from metadata
                        pages = doc.metadata.get('pages', [])
                        page_str = f"Page{' ' if len(pages) == 1 else 's '}{', '.join(map(str, pages))}"
                        
                        # Get similarity score from metadata or calculate based on position
                        score = doc.metadata.get('score', None)
                        if score is None:
                            # Fallback to position-based scoring if no actual score
                            score = max(1.0 - (idx - 1) * 0.15, 0.4)  # Decrease by 15% for each position, minimum 40%
                        
                        # Convert score to percentage
                        relevance_score = int(score * 100)
                        
                        # Create the source reference with relevance bar
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            # Get the text content directly from the source document
                            highlight_text = ""
                            # First try to get a sample of text from the page_content
                            if doc.page_content:
                                # Extract a larger sample (first 200 chars) for better matching
                                highlight_text = doc.page_content[:200]
                                # Clean up the text - remove newlines and extra spaces
                                highlight_text = ' '.join(highlight_text.split())
                                print(f"Source {idx} using page_content: {highlight_text[:50]}...")
                            
                            # If that doesn't work, try the chunk_locations as fallback
                            if not highlight_text:
                                chunk_index = doc.metadata.get('chunk_index')
                                if chunk_index is not None and chunk_index < len(st.session_state.chunk_locations):
                                    location = st.session_state.chunk_locations[chunk_index]
                                    highlight_text = location.get('text', '')
                                    # Get a larger sample for better matching
                                    if highlight_text and len(highlight_text) > 200:
                                        highlight_text = highlight_text[:200]
                                    # Clean up the text - remove newlines and extra spaces
                                    highlight_text = ' '.join(highlight_text.split())
                                    print(f"Source {idx} using chunk_location: {highlight_text[:50]}...")
                                else:
                                    print(f"Source {idx}: No chunk_index or location found")
                            
                            # Use native Streamlit button with an icon
                            if st.button(f"📄 Source {idx}: {page_str}", key=f"source_{conversation_idx}_{idx}", 
                                       on_click=handle_source_click, 
                                       args=(pages[0] if pages else 1, highlight_text)):
                                pass
                                
                            # Add custom styling for the source buttons
                            st.markdown("""
                            <style>
                                /* Style the source buttons */
                                button[data-testid*="source_"] {
                                    background-color: rgb(66, 133, 244) !important;
                                    color: white !important;
                                    border: none !important;
                                }
                                
                                button[data-testid*="source_"]:hover {
                                    background-color: rgb(41, 98, 255) !important;
                                }
                            </style>
                            """, unsafe_allow_html=True)
                        with col2:
                            st.markdown(f'''
                                <div style="display: flex; align-items: center; gap: 10px; margin-top: 5px;">
                                    <div style="
                                        background: linear-gradient(90deg, #00acee {relevance_score}%, #f0f2f6 {relevance_score}%);
                                        height: 8px;
                                        width: 100px;
                                        border-radius: 4px;
                                        margin-right: 5px;">
                                    </div>
                                    <span style="color: #666; font-size: 0.8em;">{relevance_score}% relevant</span>
                                </div>
                            ''', unsafe_allow_html=True)

def handle_source_click(page, text):
    st.session_state.target_page = page
    st.session_state.highlight_text = text
    # Increment the render key to force a re-render of the PDF
    st.session_state.pdf_render_key += 1

def render_drafty_header():
    """Render the DraftyAI header that looks like the production app."""
    # Load and encode the logo image
    import base64
    from pathlib import Path
    
    # Read the image file as bytes and encode it
    logo_path = Path("public/logo.avif")
    with open(logo_path, "rb") as f:
        logo_bytes = f.read()
    logo_b64 = base64.b64encode(logo_bytes).decode("utf-8")
    
    # Create the header HTML with the base64 encoded logo image
    header_html = f"""
    <div style="
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 20px;
        background-color: white;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 30px;
        height: 60px;
    ">
        <div style="display: flex; align-items: center;">
            <div style="width: 130px;">
                <img src="data:image/avif;base64,{logo_b64}" alt="DraftyAI Logo" style="width: 100%;">
            </div>
        </div>
        <div style="display: flex; align-items: center; gap: 30px;">
            <a href="#" style="display: flex; align-items: center; gap: 5px; text-decoration: none;">
                <span style="color: #666; font-size: 14px;">
                    <i class="fas fa-home"></i> Dashboard
                </span>
            </a>
            <a href="#" style="display: flex; align-items: center; gap: 5px; text-decoration: none;">
                <span style="color: #666; font-size: 14px;">
                    <i class="fas fa-users"></i> Clients
                </span>
            </a>
            <a href="#" style="display: flex; align-items: center; gap: 5px; text-decoration: none;">
                <span style="color: #666; font-size: 14px;">
                    <i class="fas fa-file-alt"></i> My Drafts
                </span>
            </a>
            <div style="
                display: flex;
                align-items: center;
                gap: 8px;
                background-color: #f8f9fa;
                padding: 5px 12px;
                border-radius: 20px;
            ">
                <div style="
                    width: 24px;
                    height: 24px;
                    border-radius: 50%;
                    background-color: #6c757d;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 12px;
                    color: white;
                    font-weight: bold;
                ">J</div>
                <span style="color: #666; font-size: 13px; white-space: nowrap;">Remaining Credits</span>
                <div style="
                    width: 100px;
                    height: 8px;
                    background-color: #e9ecef;
                    border-radius: 4px;
                    overflow: hidden;
                ">
                    <div style="
                        width: 33%;
                        height: 100%;
                        background-color: rgb(66, 133, 244);
                        border-radius: 4px;
                    "></div>
                </div>
                <span style="color: #666; font-size: 13px; white-space: nowrap;">250/750</span>
            </div>
        </div>
    </div>
    
    <!-- Add Font Awesome for icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
    """
    
    # Render the header
    st.markdown(header_html, unsafe_allow_html=True)

def main():
    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
    if "is_loading" not in st.session_state:
        st.session_state.is_loading = False
    if "submitted_input" not in st.session_state:
        st.session_state.submitted_input = ""
    if "current_question" not in st.session_state:
        st.session_state.current_question = ""
    if "render_key" not in st.session_state:
        st.session_state.render_key = 0
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = []
    if "chunk_locations" not in st.session_state:
        st.session_state.chunk_locations = []
    if "current_source_docs" not in st.session_state:
        st.session_state.current_source_docs = []
    if "target_page" not in st.session_state:
        st.session_state.target_page = 1
    if "highlight_text" not in st.session_state:
        st.session_state.highlight_text = ""
    if "pdf_render_key" not in st.session_state:
        st.session_state.pdf_render_key = 0
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
        
    # Add global styling for all buttons
    st.markdown("""
    <style>
        /* Global button styling */
        .stButton > button {
            background-color: rgb(66, 133, 244) !important;
            color: white !important;
            border: none !important;
        }
        
        .stButton > button:hover {
            background-color: rgb(41, 98, 255) !important;
        }
        
        /* Style the file uploader */
        .stFileUploader > div > div {
            background-color: rgb(66, 133, 244) !important;
            color: white !important;
        }
        
        /* Style the zoom and autofit buttons */
        button {
            background-color: rgb(66, 133, 244) !important;
            color: white !important;
            border: none !important;
            border-radius: 4px !important;
        }
        
        button:hover {
            background-color: rgb(41, 98, 255) !important;
        }
        
        /* Style the submit button */
        div[data-testid="stFormSubmitButton"] > button {
            background-color: rgb(66, 133, 244) !important;
            color: white !important;
            border-radius: 50% !important;
            width: 40px !important;
            height: 40px !important;
            padding: 0px !important;
            font-size: 20px !important;
            line-height: 1 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            margin: 0 auto !important;
            min-width: unset !important;
        }
        
        div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: rgb(41, 98, 255) !important;
        }
        
        /* Remove form border */
        div[data-testid="stForm"] {
            border: none !important;
            padding: 0 !important;
        }
        
        /* Adjust text area width */
        .stTextArea textarea {
            border-radius: 10px !important;
            border: 1px solid #e0e0e0 !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Render the DraftyAI header
    render_drafty_header()
    
    # Add some spacing after the header
    st.markdown("<div style='padding: 20px;'></div>", unsafe_allow_html=True)
    
    # Check if processing is complete
    if 'processing_complete' not in st.session_state or not st.session_state.processing_complete:
        # Upload PDFs - Moved from sidebar to main area using columns for proper width control
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", 
                accept_multiple_files=True,
                type=['pdf'],
                label_visibility="collapsed"
            )
            if st.button("Process", use_container_width=True):
                with st.spinner("Processing documents..."):
                    if pdf_docs:  # Only process if there are documents
                        try:
                            # Process documents and handle async operation
                            import asyncio
                            raw_text, chunk_locations = asyncio.run(process_documents(pdf_docs))
                            
                            if raw_text and chunk_locations:
                                # Store chunk locations in session state
                                st.session_state.chunk_locations = chunk_locations
                                print(f"Stored {len(chunk_locations)} chunk locations in session state")
                                # Print the first chunk location as a sample
                                if chunk_locations:
                                    print(f"Sample chunk location: {chunk_locations[0]}")
                                
                                # Get the text chunks
                                text_chunks, new_chunk_locations = get_text_chunks(raw_text, chunk_locations)
                                
                                # Create vector store
                                vectorstore = get_vectorstore(text_chunks, new_chunk_locations)
                                
                                # Check if vectorstore was created successfully
                                if vectorstore is None:
                                    raise Exception("Failed to initialize vector database. Please check that faiss-cpu is installed correctly.")
                                
                                # Create conversation chain
                                st.session_state.conversation = get_conversation_chain(vectorstore)
                                
                                st.session_state.pdf_docs = pdf_docs
                                st.session_state.processing_complete = True
                                st.success("Documents processed successfully!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"Error processing documents: {str(e)}")
                            st.session_state.processing_complete = False
    else:
        # Create columns for the document viewer and chat
        doc_col, chat_col = st.columns([1, 1], gap="medium")
        
        # Left side - PDF viewer in a container
        with doc_col:
            pdf_container = st.container(height=650)
            with pdf_container:
                pdf_doc = st.session_state.pdf_docs[0]
                pdf_doc.seek(0)
                pdf_bytes = pdf_doc.read()
                st.markdown('<div class="pdf-viewer">', unsafe_allow_html=True)
                display_pdf(pdf_bytes, highlight_text=st.session_state.highlight_text, page_num=st.session_state.target_page, key=st.session_state.pdf_render_key)
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Right side - Chat responses
        with chat_col:
            # Chat history in a fixed height container
            chat_container = st.container(height=650)
            with chat_container:
                display_chat_history()
        
        # Chat input below both columns
        if st.session_state.is_loading:
            # Display a loading animation
            loading_html = """
            <div class="loading-container" style="display: flex; justify-content: center; margin: 20px 0;">
                <div class="loading-spinner" style="
                    width: 40px;
                    height: 40px;
                    border: 5px solid rgba(66, 133, 244, 0.2);
                    border-top-color: rgb(66, 133, 244);
                    border-radius: 50%;
                    animation: spin 1s ease-in-out infinite;
                "></div>
            </div>
            <style>
                @keyframes spin {
                    to { transform: rotate(360deg); }
                }
            </style>
            <div style="text-align: center; color: #666; margin-top: 10px;">
                Thinking...
            </div>
            """
            st.markdown(loading_html, unsafe_allow_html=True)
            
            # Add a temporary user message to show what was asked
            if 'current_question' in st.session_state and st.session_state.current_question:
                st.markdown(user_template.replace("{{MSG}}", st.session_state.current_question), unsafe_allow_html=True)
        else:
            # Create a form with no border
            with st.form(key="chat_form", border=False):
                col1, col2 = st.columns([9, 1])
                with col1:
                    user_input = st.text_area("", 
                                        placeholder="Ask a question about your document...", 
                                        key="user_input",
                                        label_visibility="collapsed",
                                        height=100)
                with col2:
                    # Add some vertical spacing to align the button with the text area
                    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
                    
                    # Create a simple submit button with an arrow icon
                    submit_button = st.form_submit_button(
                        label="↑",
                        help="Submit your question"
                    )
            
                # Handle form submission
                if submit_button and user_input.strip():
                    st.session_state.submitted_input = user_input
        
        # Handle submitted input
        if st.session_state.submitted_input:
            handle_userinput(st.session_state.submitted_input)
            st.session_state.submitted_input = None
            st.rerun()

if __name__ == '__main__':
    main()
