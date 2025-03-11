css = '''
<style>
/* Remove default Streamlit spacing */
.main > div:first-child {
    padding-top: 0 !important;
}

.block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
    margin-top: -4rem !important;
    height: 100vh !important;
    overflow: hidden !important;
}

div[data-testid="stToolbar"] {
    display: none;
}

.stDeployButton {
    display: none;
}

section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

section[data-testid="stSidebar"] {
    width: 250px !important;
}

/* Main content area */
.main .block-container {
    padding: 0.5rem 1rem !important;
    max-width: none !important;
    display: flex !important;
    flex-direction: column !important;
    height: 100vh !important;
    gap: 1rem !important;
}

/* Column container styles */
[data-testid="column-container"] {
    margin-bottom: 0 !important;
}

/* Container styles */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
    background-color: #f0f2f6;
    border-radius: 0.5rem;
    padding: 0.5rem;
}

/* Base styles for chat messages */
.chat-message {
    padding: 0.75rem; 
    border-radius: 0.5rem; 
    margin-bottom: 0.75rem; 
    display: flex;
    max-width: 100%;
}

.chat-message.user {
    background-color: #2b313e
}

.chat-message.bot {
    background-color: #475063
}

.chat-message .avatar {
    width: 15%;
}

.chat-message .avatar img {
    max-width: 40px;
    max-height: 40px;
    border-radius: 50%;
    object-fit: cover;
}

.chat-message .message {
    width: 85%;
    padding: 0 0.75rem;
    color: #fff;
    font-size: 0.95rem;
}

/* Source reference styles */
.page-ref {
    cursor: pointer;
    transition: all 0.2s ease;
}

.page-ref:hover {
    opacity: 0.8;
}

/* Placeholder styles */
.placeholder-container {
    display: flex;
    justify-content: center;
    align-items: center;
    height: calc(100vh - 100px);
    background-color: #f0f2f6;
    border-radius: 10px;
    width: 100%;
    margin-top: 1rem;
}

/* Remove extra padding from columns */
[data-testid="column"] {
    padding: 0 !important;
    margin: 0 !important;
}

/* Style stVerticalBlock */
[data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
}

/* Style text input */
.stTextInput {
    margin: 0 auto !important;
    padding: 0.5rem !important;
    background: white !important;
    border-radius: 0.5rem !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    width: 80% !important;
}

.stTextInput > div {
    height: auto !important;
}

.stTextInput > div > div {
    height: auto !important;
}

.stTextInput > div > div > input {
    padding: 1rem 0.75rem !important;
    height: 5rem !important;
    line-height: 1.5 !important;
    resize: none !important;
}

/* Fix iframe height */
iframe {
    height: 100% !important;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://static.wixstatic.com/media/0e9f63_efefe52050da4ddbb4d24bee75da80cf~mv2.png/v1/fill/w_448,h_110,al_c,q_85,usm_0.66_1.00_0.01,enc_avif,quality_auto/DraftyAI%20Logo_AI%20for%20Immigration%20Lawyers.png" style="max-height: 48px; max-width: 48px; object-fit: contain; background-color: white; padding: 2px;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://openclipart.org/download/247319/abstract-user-flat-3.svg" style="max-height: 48px; max-width: 48px;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
