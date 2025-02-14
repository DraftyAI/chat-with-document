import streamlit as st
from dotenv import load_dotenv
import base64
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.llms import HuggingFaceHub
from htmlTemplates import css, bot_template, user_template
import boto3
import io
import json
import time
import re
import os
import uuid

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

def extract_text_with_textract(pdf_file, file_name):
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
                
            time.sleep(1)
        
        try:
            # Clean up the S3 file regardless of success/failure
            s3.delete_object(
                Bucket=S3_BUCKET_NAME,
                Key=unique_filename
            )
        except Exception as e:
            st.warning(f"Warning: Could not delete temporary S3 file: {str(e)}")
        
        if status == 'SUCCEEDED':
            # Get all pages
            pages = []
            
            # Get the first page
            pages.append(response)
            
            # If there are more pages, get them
            next_token = response.get('NextToken', None)
            while next_token:
                response = textract.get_document_text_detection(
                    JobId=job_id,
                    NextToken=next_token
                )
                pages.append(response)
                next_token = response.get('NextToken', None)
            
            # Extract text from all pages
            text = ""
            for page in pages:
                for item in page['Blocks']:
                    if item['BlockType'] == 'LINE':
                        text += item['Text'] + "\n"
            
            if text:
                st.write(f"Successfully extracted {len(text)} characters")
                return text
            else:
                st.warning("Textract processed the document but found no text")
                return ""
        else:
            st.error(f"Textract job failed with status: {status}")
            return ""
            
    except textract.exceptions.InvalidDocumentException:
        st.error("The document format is not supported or is corrupted")
        return ""
    except textract.exceptions.InvalidParameterException:
        st.error("The document is too large or in an invalid format")
        return ""
    except textract.exceptions.UnsupportedDocumentException:
        st.error("The document format is not supported by Textract")
        return ""
    except textract.exceptions.DocumentTooLargeException:
        st.error("The document is too large for Textract to process")
        return ""
    except textract.exceptions.ProvisionedThroughputExceededException:
        st.error("Textract throughput limit exceeded. Please try again in a moment")
        return ""
    except textract.exceptions.InternalServerError:
        st.error("AWS Textract encountered an internal error. Please try again")
        return ""
    except Exception as e:
        st.error(f"Error processing document with Textract: {str(e)}")
        return ""

def get_pdf_text(pdf_docs):
    text = ""
    chunk_locations = []
    
    for pdf in pdf_docs:
        st.write(f"Processing file: {pdf.name}")
        pdf_reader = PdfReader(pdf)
        
        # Check if any text was extracted from the document
        has_text = False
        
        # Try PyPDF2 first
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            
            # Debug information
            char_count = len(page_text) if page_text else 0
            st.write(f"Page {page_num + 1}: {char_count} characters")
            
            if page_text:
                has_text = True
                # Store the location information for this chunk
                chunk_locations.append({
                    'file': pdf.name,
                    'page': page_num + 1,
                    'text': page_text,
                    'start_char': len(text),
                    'end_char': len(text) + len(page_text)
                })
                text += page_text
        
        # If no text was found, try Textract
        if not has_text:
            st.info(f"No text found in '{pdf.name}' using standard extraction. Trying AWS Textract...")
            
            # Reset file pointer
            pdf.seek(0)
            
            # Try extracting text with Textract
            textract_text = extract_text_with_textract(pdf, pdf.name)
            
            if textract_text:
                has_text = True
                chunk_locations.append({
                    'file': pdf.name,
                    'page': 1,  # Textract doesn't provide page numbers
                    'text': textract_text,
                    'start_char': len(text),
                    'end_char': len(text) + len(textract_text)
                })
                text += textract_text
                st.success(f"Successfully extracted text from '{pdf.name}' using AWS Textract")
            else:
                st.error(f"""Could not extract text from '{pdf.name}' using either method. This might be due to:
                1. The PDF is heavily secured
                2. The image quality is too low
                3. The document contains unsupported characters or formatting""")
                return "", []
    
    return text, chunk_locations

def get_text_chunks(text, chunk_locations):
    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Calculate chunk locations
    new_chunk_locations = []
    
    for chunk in chunks:
        # Find which page this chunk belongs to
        chunk_start = text.find(chunk)
        chunk_end = chunk_start + len(chunk)
        
        # Find the page that contains this chunk
        containing_page = None
        for loc in chunk_locations:
            if (chunk_start >= loc['start_char'] and chunk_start < loc['end_char']) or \
               (chunk_end > loc['start_char'] and chunk_end <= loc['end_char']):
                containing_page = loc
                break
        
        if containing_page:
            new_chunk_locations.append({
                'file': containing_page['file'],
                'page': containing_page['page'],
                'text': chunk[:200] + "..." if len(chunk) > 200 else chunk
            })
        else:
            # Fallback if we can't find the page
            new_chunk_locations.append({
                'file': chunk_locations[0]['file'],
                'page': 1,
                'text': chunk[:200] + "..." if len(chunk) > 200 else chunk
            })
    
    return chunks, new_chunk_locations

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # Store chunk locations in metadata
    documents = [
        Document(
            page_content=chunk,
            metadata={'chunk_index': i}
        ) for i, chunk in enumerate(text_chunks)
    ]
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    # Create a retriever that includes similarity scores
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "score_threshold": 0.5,
            "k": 4
        }
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key='answer'
    )
    return conversation_chain

def display_pdf(pdf_bytes, highlight_text=None, page_num=1):
    try:
        # Try to read with PyPDF2 to validate PDF
        try:
            reader = PdfReader(io.BytesIO(pdf_bytes))
            num_pages = len(reader.pages)
        except Exception as e:
            st.error(f"PDF validation error: {str(e)}")
            return

        # Encode PDF bytes to base64
        base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
        
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
                        const container = document.getElementById('pdf-container');
                        const pages = container.children;
                        if (pageNum > 0 && pageNum <= pages.length) {{
                            const targetPage = pages[pageNum - 1];
                            if (targetPage) {{
                                targetPage.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                            }}
                        }}
                    }}

                    addEventListener('message', function(e) {{
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

                        if ('{highlight_text}') {{
                            const textContent = await page.getTextContent();
                            const textItems = textContent.items;
                            
                            textItems.forEach(function(textItem) {{
                                if (textItem.str.toLowerCase().includes('{highlight_text}'.toLowerCase())) {{
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
        
        # Add a hidden component to handle page navigation
        st.markdown(f'''
            <script>
                // Function to navigate to a specific page
                window.navigateToPage = function(pageNum) {{
                    const viewer = document.querySelector('iframe');
                    if (viewer) {{
                        viewer.contentWindow.postMessage({{
                            type: 'scrollToPage',
                            pageNum: pageNum
                        }}, '*');
                    }}
                }};
            </script>
        ''', unsafe_allow_html=True)
        
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
            
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        st.session_state.current_question = user_question
        if 'source_documents' in response:
            st.session_state.current_source_docs = response['source_documents']
        else:
            st.session_state.current_source_docs = []
        
        # Force a rerun to update the chat history
        st.rerun()

def display_chat_history():
    if not st.session_state.chat_history:
        return
    
    # Create a scrolling container for chat messages with fixed height
    chat_container = st.container(height=650)
    with chat_container:
        source_counter = 0
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                bot_message = bot_template.replace("{{MSG}}", message.content)
                
                # Add sources explanation if we have source documents
                if st.session_state.current_source_docs:
                    st.markdown("""
                        <div style="margin-top: 10px; margin-bottom: 10px; padding: 10px; border-radius: 5px; background-color: #f0f2f6;">
                            <p style="margin: 0; font-size: 0.9em;">
                                <strong>📚 Sources:</strong> Listed below in order of relevance to your question. 
                                The relevance score indicates how closely each source matches your query.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Display source references with relevance scores
                    for doc in st.session_state.current_source_docs:
                        chunk_index = doc.metadata.get('chunk_index')
                        if chunk_index is not None and chunk_index < len(st.session_state.chunk_locations):
                            location = st.session_state.chunk_locations[chunk_index]
                            page_num = location['page']
                            
                            # Get similarity score from metadata or default to a calculated one
                            score = doc.metadata.get('score', 0.0)
                            relevance_score = int(score * 100)
                            
                            st.markdown(f'''
                                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                                    <button class="page-ref" data-page="{page_num}" 
                                            style="min-width: 150px;">
                                        📄 Page {page_num}
                                    </button>
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
                
                st.markdown(bot_message, unsafe_allow_html=True)
                
                # Display source references
                if st.session_state.current_source_docs:
                    st.markdown("### Sources:", unsafe_allow_html=True)
                    for i, doc in enumerate(st.session_state.current_source_docs, 1):
                        chunk_index = doc.metadata.get('chunk_index')
                        if chunk_index is not None and chunk_index < len(st.session_state.chunk_locations):
                            location = st.session_state.chunk_locations[chunk_index]
                            page_num = location['page']
                            source_counter += 1
                            
                            if st.button(f"📄 Source {i}: Page {page_num}", 
                                       key=f"source_{source_counter}"):
                                st.session_state.target_page = page_num
                                st.rerun()
                            
                            st.markdown(f"""
                                <div style="margin-left: 20px; margin-top: 5px; font-size: 0.9em; color: #666;">
                                    "{location['text']}"
                                </div>
                            """, unsafe_allow_html=True)

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
        st.session_state.highlight_text = None
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
                        # get pdf text
                        raw_text, chunk_locations = get_pdf_text(pdf_docs)
                        st.session_state.chunk_locations = chunk_locations
                        
                        # get the text chunks
                        text_chunks, new_chunk_locations = get_text_chunks(raw_text, chunk_locations)
                        
                        # create vector store
                        vectorstore = get_vectorstore(text_chunks)
                        
                        # create conversation chain
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
