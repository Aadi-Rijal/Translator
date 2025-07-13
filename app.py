import streamlit as st
from pathlib import Path
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import torch
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Translator",
    page_icon="🇳🇵",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Minimal, clean CSS
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {padding: 2rem 1rem;}
    
    /* Clean typography */
    .main-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1rem;
        color: #666;
        font-weight: 400;
    }
    
    /* Card container */
    .stContainer > div {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.06);
        border: 1px solid #f0f0f0;
        margin-bottom: 1.5rem;
    }
    
    /* Custom textarea */
    .stTextArea > div > div > textarea {
        border-radius: 12px !important;
        border: 1px solid #e0e0e0 !important;
        padding: 1rem !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
        resize: vertical !important;
        min-height: 120px !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #007bff !important;
        box-shadow: 0 0 0 3px rgba(0,123,255,0.1) !important;
    }
    
    /* Action buttons */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.5rem !important;
        border: none !important;
        transition: all 0.2s ease !important;
        font-size: 0.95rem !important;
        width: 100% !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(45deg, #007bff, #0056b3) !important;
        color: white !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(0,123,255,0.3) !important;
    }
    
    .stButton > button[kind="secondary"] {
        background: #f8f9fa !important;
        color: #666 !important;
        border: 1px solid #e0e0e0 !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: #e9ecef !important;
    }
    
    /* Progress indicator */
    .progress-indicator {
        background: linear-gradient(45deg, #fff3cd, #ffeaa7);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        color: #856404;
        font-weight: 500;
        margin: 1rem 0;
        border: 1px solid #ffeaa7;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #333;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Character count */
    .char-count {
        text-align: right;
        color: #999;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 0.5rem;
        }
        
        .main-title {
            font-size: 1.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the transformer model and return model, device, seq_len, and config"""
    try:
        from config import get_config
        #from dataset import latest_weights_file_path
        from model import build_transformer
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = get_config()
        
        model = build_transformer(
            30000, 30000, 
            config["seq_len"], 
            config['seq_len'], 
            d_model=config['d_model']
        ).to(device)
        
        #model_filename = latest_weights_file_path(config)
        # if not os.path.exists(model_filename):
        #     raise FileNotFoundError(f"Model file not found: {model_filename}")
        model_filename = hf_hub_download(repo_id="Aadi-Rijal/translator-transformer", filename="tmodel_01_264000.pt")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        model.eval()
        
        logger.info(f"Model loaded successfully on {device}")
        return model, device, config['seq_len'], config
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

@st.cache_resource
def get_tokenizers(src_lang, dest_lang, _config):
    """Load tokenizers for source and destination languages"""
    try:
        tokenizer_src_path = Path(_config['tokenizer_file'].format(src_lang))
        tokenizer_tgt_path = Path(_config['tokenizer_file'].format(dest_lang))
        
        if not tokenizer_src_path.exists():
            raise FileNotFoundError(f"Source tokenizer not found: {tokenizer_src_path}")
        if not tokenizer_tgt_path.exists():
            raise FileNotFoundError(f"Target tokenizer not found: {tokenizer_tgt_path}")
        
        tokenizer_src = Tokenizer.from_file(str(tokenizer_src_path))
        tokenizer_tgt = Tokenizer.from_file(str(tokenizer_tgt_path))
        
        return tokenizer_src, tokenizer_tgt
        
    except Exception as e:
        logger.error(f"Error loading tokenizers: {str(e)}")
        raise e

def translate_text(text, model, device, seq_len, config):
    """Translate Nepali text to English"""
    try:
        from validate import greedy_decode
        
        src_lang = 'np'
        dest_lang = 'en'
        token_lang_src = "NE"
        token_lang_tgt = "EN"
        
        tokenizer_src, tokenizer_tgt = get_tokenizers(src_lang, dest_lang, config)
        
        with torch.no_grad():
            source = tokenizer_src.encode(text)

            source = torch.cat([
                torch.tensor([tokenizer_src.token_to_id(f'[{token_lang_src}]')], dtype=torch.int64), 
                torch.tensor(source.ids, dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id('[EOS]')], dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id('[PAD]')] * (seq_len - len(source.ids) - 2), dtype=torch.int64)
            ], dim=0).to(device)
            
            source_mask = (source != tokenizer_src.token_to_id('[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)
            
            model_out = greedy_decode(
                model, token_lang_src, source, source_mask, 
                tokenizer_src, tokenizer_tgt, 300, device
            )
            
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            model_out_text = model_out_text.replace('[EOS]', '').replace('[PAD]', '').strip()
            
            return model_out_text
            
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        raise e

def main():
    # Initialize session state
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'translation_history' not in st.session_state:
        st.session_state.translation_history = []
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""
    if 'output_text' not in st.session_state:
        st.session_state.output_text = ""
    if 'translation_direction' not in st.session_state:
        st.session_state.translation_direction = "np_to_en"
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">Translator</h1>
        <p class="subtitle">Breaking language barriers with every text.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("🔄 Loading translation model..."):
            try:
                model, device, seq_len, config = load_model()
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.seq_len = seq_len
                st.session_state.config = config
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to load model: {str(e)}")
                st.stop()
    
    # Main translator container
    with st.container():
        direction_options = {
            "🇳🇵 → 🇪🇳": "np_to_en",
            "🇪🇳 → 🇳🇵": "en_to_np"
        }

        selected = st.selectbox("",
            list(direction_options.keys()),
            index=0 if st.session_state.translation_direction == "np_to_en" else 1
        )

        selected_direction = direction_options[selected]
        if selected_direction != st.session_state.translation_direction:
            st.session_state.translation_direction = selected_direction
            st.session_state.input_text = ""
            st.session_state.output_text = ""
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)

        
        # Check if English to Nepali is selected
        if st.session_state.translation_direction == "en_to_np":
            st.markdown("""
            <div class="progress-indicator">
                🚧 English to Nepali translation is coming soon!
            </div>
            """, unsafe_allow_html=True)
        else:
            # Get current input text
            current_input = st.session_state.get('input_text', '')
            
            input_text = st.text_area(
                "",
                value=current_input,
                height=120,
                placeholder="पाठ प्रविष्ट गर्नुहोस्...",
                max_chars=500,
                label_visibility="collapsed",
                key="input_area"
            )
            
            # Update session state
            st.session_state.input_text = input_text
            
            # Character count
            if input_text:
                char_count = len(input_text)
                st.markdown(f"<div class='char-count'>{char_count}/500 characters</div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Action buttons
            col1, col2 = st.columns([1, 1])
            
            with col1:
                translate_btn = st.button("🔄 Translate", type="primary", key="translate_btn")
            
            with col2:
                clear_btn = st.button("🗑️ Clear", type="secondary", key="clear_btn")
            
            # Handle button clicks
            if clear_btn:
                st.session_state.input_text = ""
                st.session_state.output_text = ""
                st.rerun()
            
            # Translation processing
            if translate_btn:
                if input_text and input_text.strip():
                    try:
                        with st.spinner("🔄 Translating..."):
                            translated_text = translate_text(
                                input_text.strip(), 
                                st.session_state.model, 
                                st.session_state.device, 
                                st.session_state.seq_len, 
                                st.session_state.config
                            )
                        
                        st.session_state.output_text = translated_text
                        
                        # Add to history
                        st.session_state.translation_history.append({
                            'input': input_text.strip(),
                            'output': translated_text,
                            'direction': st.session_state.translation_direction,
                            'timestamp': time.time()
                        })
                        
                        # Keep only last 10 translations
                        if len(st.session_state.translation_history) > 10:
                            st.session_state.translation_history.pop(0)
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Translation failed: {str(e)}")
                else:
                    st.warning("⚠️ Please enter some text to translate")
            
            # Output section
            if st.session_state.output_text:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">✅ Translation Result</div>', unsafe_allow_html=True)
                
                st.text_area(
                    "",
                    value=st.session_state.output_text,
                    height=120,
                    disabled=True,
                    label_visibility="collapsed",
                    key="output_area"
                )

    
    # Translation history
    if st.session_state.translation_history:
        st.markdown("<br>", unsafe_allow_html=True)
        
        with st.expander("📚 Translation History", expanded=False):
            for i, trans in enumerate(reversed(st.session_state.translation_history[-5:])):
                st.markdown(f"**{i+1}. Input:** {trans['input'][:100]}{'...' if len(trans['input']) > 100 else ''}")
                st.markdown(f"**Output:** {trans['output']}")
                st.markdown("---")
            
            if len(st.session_state.translation_history) > 5:
                st.info(f"📊 Showing latest 5 of {len(st.session_state.translation_history)} translations")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🗑️ Clear History", type="secondary", key="clear_history_btn"):
                    st.session_state.translation_history = []
                    st.rerun()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #999; font-size: 0.85rem; padding-top: 2rem; margin-top: 2rem; border-top: 1px solid #f0f0f0;">
        Made to help everyone.
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()