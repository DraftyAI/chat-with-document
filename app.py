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

# Set page config to minimize default padding
st.set_page_config(
    page_title="Chat with multiple PDFs",
    page_icon=":books:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
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

async def extract_text_with_textract(pdf_file, file_name):
    # Initialize Textract client
    textract = boto3.client('textract')
    s3 = boto3.client('s3')
    
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

def display_pdf(pdf_bytes, highlight_text=None, page_num=1):
    try:
        # Add logging for highlight_text
        print(f"display_pdf called with highlight_text: {highlight_text}")
        
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

                    async function renderPage(pdf, pageNum, container) {{
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

                        // Only highlight if highlight_text is provided and not empty
                        const highlightText = '{highlight_text}';
                        console.log('Page', pageNum, 'checking highlight text:', highlightText);
                        console.log('Is highlight text empty?', !highlightText || highlightText.trim() === '');
                        console.log('Highlight text type:', typeof highlightText);
                        console.log('Highlight text length:', highlightText ? highlightText.length : 0);
                        
                        // Check that highlight text is not empty and not the string "None"
                        if (highlightText && highlightText.trim() !== '' && highlightText !== 'None') {{
                            console.log('Highlighting text on page', pageNum, ':', highlightText);
                            const textContent = await page.getTextContent();
                            const textItems = textContent.items;
                            
                            console.log('Found', textItems.length, 'text items on page', pageNum);
                            
                            // Keep track of whether we found any matches
                            let foundMatch = false;
                            let matchCount = 0;
                            
                            // Break the highlight text into words for more flexible matching
                            const highlightWords = highlightText.toLowerCase().split(/\s+/).filter(word => word.length > 3);
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
                                if (!shouldHighlight && itemText.includes(highlightText.toLowerCase())) {{
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

                    async function renderAllPages() {{
                        const container = document.getElementById('pdf-container');
                        container.innerHTML = '';
                        
                        for (let pageNum = 1; pageNum <= pdfDocument.numPages; pageNum++) {{
                            await renderPage(pdfDocument, pageNum, container);
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
                            status.textContent = 'Rendering PDF...';
                            
                            // Calculate initial scale based on first page
                            const firstPage = await pdfDocument.getPage(1);
                            const container = document.getElementById('pdf-container');
                            defaultScale = calculateFitScale(firstPage, container);
                            currentScale = defaultScale;
                            
                            await renderAllPages();
                            status.textContent = `PDF loaded successfully (${{pdfDocument.numPages}} pages)`;
                            
                            // Hide status after 5 seconds
                            setTimeout(() => {{
                                status.style.opacity = '0';
                                setTimeout(() => {{
                                    status.style.display = 'none';
                                    status.style.opacity = '1';
                                }}, 300); // Wait for fade out animation
                            }}, 5000);
                            
                            scrollToPage({page_num});
                            
                        }} catch (error) {{
                            document.getElementById('error').style.display = 'block';
                            document.getElementById('error').textContent = 'Error: ' + error.message;
                            console.error('Error:', error);
                        }}
                    }}

                    // Make zoom functions globally available
                    window.zoomIn = zoomIn;
                    window.zoomOut = zoomOut;
                    window.resetZoom = resetZoom;

                    loadPDF();

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
                            
                            if st.button(f"📄 Source {idx}: {page_str}", key=f"source_{conversation_idx}_{idx}"):
                                st.session_state.target_page = pages[0] if pages else 1
                                st.session_state.highlight_text = highlight_text
                                print(f"Button clicked: Setting highlight_text to: {highlight_text[:30]}...")
                                st.rerun()
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

def main():
    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = None
    if "chunk_locations" not in st.session_state:
        st.session_state.chunk_locations = None
    if "highlight_text" not in st.session_state:
        st.session_state.highlight_text = ""  # Initialize as empty string instead of None
    if "target_page" not in st.session_state:
        st.session_state.target_page = 1
    if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
    if "submitted_input" not in st.session_state:
        st.session_state.submitted_input = None

    # Upload PDFs
    with st.sidebar:
        st.markdown("### Your documents")
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
                    st.warning("Please upload at least one PDF document.")

    # Create the main layout based on whether documents are processed
    if st.session_state.pdf_docs and st.session_state.processing_complete:
        # Create columns for the document viewer and chat
        doc_col, chat_col = st.columns([1, 1], gap="medium")
        
        # Left side - PDF viewer in a container
        with doc_col:
            pdf_container = st.container(height=650)
            with pdf_container:
                pdf_doc = st.session_state.pdf_docs[0]
                pdf_doc.seek(0)
                pdf_bytes = pdf_doc.read()
                display_pdf(pdf_bytes, highlight_text=st.session_state.highlight_text, page_num=st.session_state.target_page)
        
        # Right side - Chat responses
        with chat_col:
            # Chat history in a fixed height container
            chat_container = st.container(height=650)
            with chat_container:
                display_chat_history()
        
        # Chat input below both columns
        def submit():
            if st.session_state.user_input.strip():
                st.session_state.submitted_input = st.session_state.user_input
                st.session_state.user_input = ""

        # Create a three-column layout with the middle column taking 80% width
        _, input_col, _ = st.columns([1, 4, 1])
        with input_col:
            st.markdown("""
                <style>
                    .stTextArea textarea {
                        height: 100px !important;
                        min-height: 100px !important;
                        max-height: 100px !important;
                        padding: 10px !important;
                        font-size: 1em !important;
                        border-radius: 10px !important;
                        border: 1px solid #ccc !important;
                        resize: none !important;
                    }
                    .stTextArea div[data-baseweb="textarea"] {
                        height: auto !important;
                        margin-bottom: 30px !important;  /* Add space for the hint */
                    }
                    .stTextArea label {
                        display: none !important;
                    }
                    /* Style the command hint text */
                    small[data-testid="stChatInputStatus"] {
                        position: absolute !important;
                        bottom: 15px !important;
                        right: 15px !important;
                        padding: 4px 8px !important;
                        background: rgba(255, 255, 255, 0.9) !important;
                        border-radius: 4px !important;
                        font-size: 0.8em !important;
                        color: #666 !important;
                        margin: 0 !important;
                    }
                </style>
            """, unsafe_allow_html=True)
            
            st.text_area("", 
                        placeholder="Ask a question about your document...", 
                        key="user_input",
                        on_change=submit,
                        label_visibility="collapsed",
                        height=100)
        
        # Handle submitted input
        if st.session_state.submitted_input:
            handle_userinput(st.session_state.submitted_input)
            st.session_state.submitted_input = None
            st.rerun()
    else:
        # Show full-width placeholder when no documents are processed
        st.markdown(
            """
            <div class="placeholder-container">
                <div style="text-align: center; color: #666;">
                    <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" 
                         stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                        <line x1="12" y1="18" x2="12" y2="12"></line>
                        <line x1="9" y1="15" x2="15" y2="15"></line>
                    </svg>
                    <p style="margin-top: 10px;">Upload a PDF to begin</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == '__main__':
    main()
