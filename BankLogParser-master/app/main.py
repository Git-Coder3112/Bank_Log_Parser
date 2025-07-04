import streamlit as st
from streamlit_option_menu import option_menu
import os
from PIL import Image
import base64
import pandas as pd
import numpy as np
import torch
import plotly.express as px
from datetime import datetime, timedelta
from utils.file_handlers import FileHandler
from utils.transaction_tagger import TransactionTagger
from utils.fraud_detector import FraudDetector

def ensure_native_df(df):
    """Ensure the DataFrame is a native pandas DataFrame."""
    if hasattr(df, 'to_native'):
        return df.to_native()
    return df

def set_page_config():
    st.set_page_config(
        page_title="Bank Log Analyzer",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def set_custom_css():
    st.markdown("""
    <style>
        .main-header {font-size: 24px; font-weight: 700; color: #1e3a8a;}
        .sub-header {font-size: 18px; font-weight: 600; color: #1e40af;}
        .file-uploader {border: 2px dashed #3b82f6; border-radius: 10px; padding: 20px;}
        .stButton>button {background-color: #1e40af; color: white; border-radius: 5px;}
        .stButton>button:hover {background-color: #1e3a8a;}
        .success-msg {color: #10b981; font-weight: 500;}
        .error-msg {color: #ef4444; font-weight: 500;}
    </style>
    """, unsafe_allow_html=True)

def display_header():
    st.title("Bank Statement Analyzer")
    st.caption("Upload your bank statements and gain valuable insights")

def main():
    set_page_config()
    set_custom_css()
    
    
    if 'use_ml' not in st.session_state:
        st.session_state.use_ml = True
        
    
    if 'tagger' not in st.session_state:
        with st.spinner('Loading transaction tagger with BERT... (this may take a minute)'):
            try:
                use_gpu = torch.cuda.is_available()
                if use_gpu:
                    st.sidebar.info("🎮 GPU acceleration is available and will be used")
                else:
                    st.sidebar.warning("⚠️ No GPU found. Using CPU (tagging may be slower)")
                
                st.session_state.tagger = TransactionTagger(use_gpu=use_gpu)
                
                if st.session_state.tagger.nlp is None:
                    st.sidebar.error("❌ Failed to load BERT model. Falling back to rule-based tagging.")
                else:
                    st.sidebar.success("✅ BERT model loaded successfully")
                    
            except Exception as e:
                st.error(f"Error initializing transaction tagger: {str(e)}")
                st.session_state.tagger = TransactionTagger(use_gpu=False)
                st.session_state.tagger.nlp = None  # Force rule-based fallback
    
    
    if 'fraud_detector' not in st.session_state:
        with st.spinner('Initializing fraud detection system...'):
            try:
                st.session_state.fraud_detector = FraudDetector(contamination=0.02)
                st.sidebar.success("✅ Fraud detection system ready")
            except Exception as e:
                st.error(f"Error initializing fraud detector: {str(e)}")
    
    
    if 'auto_tag' not in st.session_state:
        st.session_state.auto_tag = False
    if 'tag_confidence_threshold' not in st.session_state:
        st.session_state.tag_confidence_threshold = 0.7
    
    
    with st.sidebar:
        st.title("Navigation & Settings")
        
       
        st.subheader("Processing Mode")
        use_ml = st.toggle("Use AI-Powered Processing", 
                         value=st.session_state.use_ml,
                         help="Enable for better accuracy with complex statements")
        
        if use_ml != st.session_state.use_ml:
            st.session_state.use_ml = use_ml
            st.rerun()
            
        if use_ml:
            st.info("ℹ️ AI mode: Using LayoutLM for better accuracy")
        else:
            st.info("ℹ️ Standard mode: Using rule-based processing")
            
        
        st.markdown("---")
        st.subheader("Transaction Tagging")
        
        
        auto_tag = st.toggle("Auto-tag transactions",
                           value=st.session_state.auto_tag,
                           help="Automatically tag transactions with categories")
        
        if auto_tag != st.session_state.auto_tag:
            st.session_state.auto_tag = auto_tag
            st.rerun()
            
        if st.session_state.auto_tag:
            
            conf_threshold = st.slider(
                "Tagging Confidence Threshold",
                min_value=0.5,
                max_value=1.0,
                value=st.session_state.tag_confidence_threshold,
                step=0.05,
                help="Higher values mean more confident tags, but fewer transactions will be tagged"
            )
            if conf_threshold != st.session_state.tag_confidence_threshold:
                st.session_state.tag_confidence_threshold = conf_threshold
                st.rerun()
            
        st.markdown("---")
        
        
        selected = option_menu(
            menu_title=None,
            options=["Home", "Dashboard", "Settings"],
            icons=["house", "speedometer2", "gear"],
            default_index=0,
        )
    
    if selected == "Home":
        display_header()
        st.markdown("### Upload Your Bank Statement")
        
        
        uploaded_file = st.file_uploader(
            "",
            type=["pdf", "xlsx", "xls", "png", "jpg", "jpeg"],
            accept_multiple_files=False,
            help="Upload your bank statement (PDF, Excel, or Image)",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            st.success(f"Successfully uploaded: {uploaded_file.name}")
            
            with st.spinner("Processing file..."):
                try:
                    
                    file_handler = FileHandler(use_ml=st.session_state.use_ml)
                    file_info = file_handler.handle_file_upload(uploaded_file)
                    
                    
                    st.session_state.file_info = file_info
                    
                    
                    st.subheader("File Information")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("File Name", file_info['filename'])
                        st.metric("File Type", file_info['type'])
                    with col2:
                        st.metric("File Size", f"{file_info['size'] / 1024:.2f} KB")
                    
                    if file_info.get('processed', False):
                        st.success("File processed successfully!")
                        
                        
                        st.subheader("Extracted Transactions")
                        if not file_info['data'].empty:
                           
                            display_df = file_info['data'].copy()
                            if hasattr(display_df, 'to_native'):
                                display_df = display_df.to_native()
                            
                            
                            display_df = display_df.copy()
                            
                            
                            if 'amount' in display_df.columns:
                                display_df['amount_display'] = display_df['amount'].apply(
                                    lambda x: f"₹{float(x):,.2f}" if pd.notnull(x) and str(x).strip() else ""
                                )
                            if 'closing_balance' in display_df.columns:
                                display_df['closing_balance_display'] = display_df['closing_balance'].apply(
                                    lambda x: f"₹{float(x):,.2f}" if pd.notnull(x) and str(x).strip() else ""
                                )
                            
                           
                            columns_to_display = ['datetime', 'type', 'description', 'amount_display', 'location', 'closing_balance_display']
                            existing_columns = [col for col in columns_to_display if col in display_df.columns]
                            
                           
                            column_mapping = {
                                'amount_display': 'amount',
                                'closing_balance_display': 'closing_balance'
                            }
                            processing_columns = [column_mapping.get(col, col) for col in existing_columns]
                            
                            
                            if 'datetime' in display_df.columns:
                                display_df['datetime'] = pd.to_datetime(display_df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
                            
                            
                            if st.session_state.auto_tag and 'description' in display_df.columns:
                                with st.spinner('Tagging transactions...'):
                                    
                                    df_to_tag = display_df[processing_columns].copy()
                                    
                                   
                                    if 'amount' in df_to_tag.columns:
                                        if df_to_tag['amount'].dtype == 'object':
                                            df_to_tag['amount'] = pd.to_numeric(
                                                df_to_tag['amount'].astype(str).str.replace('[^\d.-]', '', regex=True),
                                                errors='coerce'
                                            )
                                    
                                   
                                    st.sidebar.write("Debug - Sample data being processed:")
                                    st.sidebar.dataframe(df_to_tag[['description', 'amount']].head())
                                    
                                    
                                    tagged_df = st.session_state.tagger.tag_dataframe(
                                        df_to_tag,
                                        description_col='description',
                                        amount_col='amount',
                                        use_ml=st.session_state.use_ml
                                    )
                                    
                                    
                                    st.sidebar.write("Debug - Sample tags:")
                                    st.sidebar.dataframe(tagged_df[['description', 'tag', 'tag_confidence']].head())
                                    
                                    
                                    mask = tagged_df['tag_confidence'] >= st.session_state.tag_confidence_threshold
                                    tagged_df.loc[~mask, 'tag'] = 'uncategorized'
                                    
                                    
                                    display_df['tag'] = tagged_df['tag']
                                    
                                    
                                    if 'tag' not in existing_columns:
                                        existing_columns.append('tag')
                            
                           
                            with st.spinner('Analyzing transactions for potential fraud...'):
                                try:
                                   
                                    fraud_df = display_df[processing_columns].copy()
                                    
                                    
                                    if 'amount' in fraud_df.columns:
                                        if fraud_df['amount'].dtype == 'object':
                                            fraud_df['amount'] = pd.to_numeric(
                                                fraud_df['amount'].astype(str).str.replace('[^\d.-]', '', regex=True),
                                                errors='coerce'
                                            ).fillna(0)
                                    
                                    
                                    if 'datetime' in fraud_df.columns:
                                        fraud_df['date'] = pd.to_datetime(fraud_df['datetime'])
                                    elif 'date' not in fraud_df.columns:
                                        fraud_df['date'] = pd.Timestamp.now() - pd.Timedelta(days=len(fraud_df)-1)
                                    
                                    
                                    st.session_state.fraud_detector.fit(fraud_df)
                                    
                                    
                                    predictions, explanations = st.session_state.fraud_detector.predict(fraud_df, return_explanations=True)
                                    
                                  
                                    display_df = display_df.copy()
                                    
                                   
                                    display_df['anomaly_score'] = [expl.anomaly_score for expl in explanations]
                                    display_df['is_anomaly'] = [1 if expl.is_fraud else 0 for expl in explanations]
                                    
                                    if 'fraud_explanations' not in st.session_state:
                                        st.session_state.fraud_explanations = {}
                                    for idx, expl in enumerate(explanations):
                                        if expl.is_fraud:
                                            st.session_state.fraud_explanations[idx] = expl.explanation_text
                                    
                                   
                                    if 'anomaly_score' not in existing_columns:
                                        existing_columns.append('anomaly_score')
                                    if 'is_anomaly' not in existing_columns:
                                        existing_columns.append('is_anomaly')
                                    
                                   
                                    display_df['anomaly_score'] = pd.to_numeric(display_df.get('anomaly_score', 0), errors='coerce').fillna(0)
                                    display_df['is_anomaly'] = pd.to_numeric(display_df.get('is_anomaly', 0), errors='coerce').fillna(0).astype(int)
                                    
                                   
                                    print("\nAnomaly values:", display_df['is_anomaly'].value_counts())
                                    if not display_df[display_df['is_anomaly'] == 1].empty:
                                        print("Sample anomaly values:", display_df[display_df['is_anomaly'] == 1][['description', 'amount', 'is_anomaly']].head())
                                    
                                   
                                    if 'anomaly_score' not in display_df.columns:
                                        display_df['anomaly_score'] = 0.0
                                    if 'is_anomaly' not in display_df.columns:
                                        display_df['is_anomaly'] = False
                                
                                except Exception as e:
                                    st.error(f"Error during fraud detection: {str(e)}")
                                   
                                    if 'anomaly_score' not in display_df.columns:
                                        display_df['anomaly_score'] = 0.0
                                    if 'is_anomaly' not in display_df.columns:
                                        display_df['is_anomaly'] = False
                            
                            
                            st.subheader("Transaction Summary")
                            
                           
                            df_stats = ensure_native_df(file_info['data'].copy())
                            
                            
                            if 'amount' in df_stats.columns:
                                df_stats['amount'] = pd.to_numeric(df_stats['amount'], errors='coerce')
                            
                            
                            total_credit = df_stats[df_stats['type'] == 'credit']['amount'].sum() if 'type' in df_stats.columns else 0
                            total_debit = df_stats[df_stats['type'] == 'debit']['amount'].sum() if 'type' in df_stats.columns else 0
                            net_balance = total_credit - total_debit
                            
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Credit", f"₹{total_credit:,.2f}" if total_credit > 0 else "₹0.00")
                            with col2:
                                st.metric("Total Debit", f"₹{total_debit:,.2f}" if total_debit > 0 else "₹0.00")
                            with col3:
                                st.metric("Net Balance", f"₹{net_balance:,.2f}")
                            
                            st.markdown("---")  
                            
                            
                            if not display_df.empty:
                                
                                def style_dataframe(row):
                                    styles = [''] * len(row)
                                    
                                    try:
                                        
                                        if 'is_anomaly' in display_df.columns and 'is_anomaly' in row.index:
                                            
                                            anomaly_value = int(float(row['is_anomaly'])) if pd.notna(row['is_anomaly']) else 0
                                            
                                            
                                            if anomaly_value == 1:
                                                
                                                styles = ['background-color: #FFEBEE; color: #B71C1C;'] * len(row)
                                                
                                               
                                                anomaly_index = display_df.columns.get_loc('is_anomaly')
                                                if 0 <= anomaly_index < len(styles):
                                                    styles[anomaly_index] = 'background-color: #FFCDD2; color: #B71C1C; font-weight: bold;'
                                        
                                        
                                        if 'tag' in display_df.columns and 'tag' in row and pd.notna(row['tag']):
                                            tag = str(row['tag']).lower()
                                            colors = {
                                                'food': {'bg': '#FFE5E5', 'text': '#B71C1C'},
                                                'shopping': {'bg': '#E5F5FF', 'text': '#0D47A1'},
                                                'transport': {'bg': '#E5FFE5', 'text': '#1B5E20'},
                                                'bills': {'bg': '#FFF0E5', 'text': '#E65100'},
                                                'groceries': {'bg': '#F5E5FF', 'text': '#4A148C'},
                                                'travel': {'bg': '#FFFFE5', 'text': '#F57F17'},
                                                'health': {'bg': '#FFE5F5', 'text': '#880E4F'},
                                                'salary': {'bg': '#E5FFE5', 'text': '#1B5E20'},
                                                'investment': {'bg': '#E5F5FF', 'text': '#0D47A1'},
                                                'transfer': {'bg': '#F0F0F0', 'text': '#212121'},
                                                'other': {'bg': '#F5F5F5', 'text': '#212121'},
                                                'uncategorized': {'bg': '#FFFFFF', 'text': '#9E9E9E'}
                                            }
                                            color = colors.get(tag, colors['uncategorized'])
                                            tag_index = display_df.columns.get_loc('tag')
                                            if 0 <= tag_index < len(styles):
                                                styles[tag_index] = f'background-color: {color["bg"]}; color: {color["text"]}'
                                    except Exception as e:
                                        print(f"Error applying styles: {str(e)}")
                                    
                                    return styles
                                
                                
                                columns_to_remove = []
                                if 'closing_balance_display' in display_df.columns and 'closing_balance' in display_df.columns:
                                    columns_to_remove.append('closing_balance')
                                if 'date' in display_df.columns and 'datetime' in display_df.columns:
                                    columns_to_remove.append('date')
                                if 'time' in display_df.columns and 'datetime' in display_df.columns:
                                    columns_to_remove.append('time')
                                if 'amount_display' in display_df.columns and 'amount' in display_df.columns:
                                    columns_to_remove.append('amount')
                                
                                
                                display_df = display_df.loc[:, ~display_df.columns.duplicated()]
                                
                                
                                preferred_order = [
                                    'datetime', 'type', 'description', 'amount_display', 'tag',
                                    'closing_balance_display', 'location', 'category', 'reference', 'balance'
                                ]
                                
                                
                                columns_to_show = [col for col in preferred_order 
                                                if col in display_df.columns 
                                                and col not in columns_to_remove]
                                
                                
                                remaining_cols = [col for col in display_df.columns 
                                                if col not in columns_to_show 
                                                and col not in ['anomaly_score', 'is_anomaly']
                                                and col not in columns_to_remove
                                                and not any(col.startswith(prefix) for prefix in ['closing_balance_', 'date_'])]
                                
                                columns_to_show.extend(remaining_cols)
                                
                                
                                if 'is_anomaly' not in display_df.columns:
                                    display_df['is_anomaly'] = 0
                                else:
                                    
                                    display_df['is_anomaly'] = pd.to_numeric(display_df['is_anomaly'], errors='coerce').fillna(0).astype(int)
                                    
                                if 'anomaly_score' not in display_df.columns:
                                    display_df['anomaly_score'] = 0.0
                                
                                
                                print("\nColumn types:")
                                print(display_df.dtypes)
                                print("\nSample is_anomaly values:")
                                print(display_df['is_anomaly'].value_counts())
                                
                                
                                if 'anomaly_score' in display_df.columns:
                                    columns_to_show.append('anomaly_score')
                                if 'is_anomaly' in display_df.columns:
                                    columns_to_show.append('is_anomaly')
                                
                                
                                styled_df = display_df[columns_to_show].style.apply(style_dataframe, axis=1)
                                
                               
                                st.markdown("""
                                <style>
                                    .stDataFrame {
                                        border: 2px solid #000000 !important;
                                        border-collapse: collapse !important;
                                        border-radius: 4px !important;
                                        overflow: hidden !important;
                                    }
                                    .stDataFrame th, .stDataFrame td {
                                        border: 1px solid #000000 !important;
                                        padding: 8px 12px !important;
                                        text-align: left !important;
                                    }
                                    .stDataFrame th {
                                        background-color: #F0F0F0 !important;
                                        font-weight: 600 !important;
                                        border-bottom: 2px solid #000000 !important;
                                    }
                                    .stDataFrame tr:hover {
                                        background-color: #F9F9F9 !important;
                                    }
                                    .stDataFrame tr:nth-child(even) {
                                        background-color: #FFFFFF !important;
                                    }
                                    .stDataFrame tr:nth-child(odd) {
                                        background-color: #F8F8F8 !important;
                                    }
                                    /* Force anomaly cell styling */
                                    .stDataFrame td[data-testid='stDataFrameCell'] {
                                        background-color: inherit !important;
                                    }
                                </style>
                                """, unsafe_allow_html=True)
                                
                                st.dataframe(
                                    styled_df,
                                    use_container_width=True,
                                    hide_index=True
                                )
                                                                
                                if 'is_anomaly' in display_df.columns and display_df['is_anomaly'].any():
                                    st.warning("⚠️ **Potential Fraudulent Transactions Detected**")
                                    
                                    
                                    st.subheader("Suspicious Transactions")
                                    
                                  
                                    suspicious = display_df[display_df['is_anomaly'] == 1].copy()
                                    
                                   
                                    for idx, row in suspicious.iterrows():
                                        with st.expander(f"Suspicious transaction: {row.get('description', 'N/A')} - ₹{row.get('amount', 0):,.2f}"):
                                            st.markdown(f"**Date:** {row.get('datetime', 'N/A')}")
                                            st.markdown(f"**Amount:** ₹{row.get('amount', 0):,.2f}")
                                            st.markdown(f"**Anomaly Score:** {row.get('anomaly_score', 0):.2f}")
                                            st.markdown("**Reason for flagging:**")
                                            
                                            explanation = st.session_state.fraud_explanations.get(idx, "No explanation available")
                                            
                                            if "Recommendation:" in explanation:
                                                explanation = explanation.split("Recommendation:")[0].strip()
                                            st.info(explanation)
                                    
                                
                                    if 'tag' in display_df.columns:
                                        st.subheader("Anomaly Distribution by Category")
                                        try:
                                            
                                            anomalies = display_df[display_df['is_anomaly'] == 1]
                                            if not anomalies.empty and 'tag' in anomalies.columns:
                                                anomaly_dist = anomalies['tag'].value_counts().reset_index()
                                                anomaly_dist.columns = ['Category', 'Count']
                                                
                                                if not anomaly_dist.empty:
                                                    fig = px.pie(
                                                        anomaly_dist,
                                                        values='Count',
                                                        names='Category',
                                                        title='Anomalies by Category',
                                                        hole=0.4
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)
                                            else:
                                                st.info("No categorized anomalies to display")
                                        except Exception as e:
                                            st.warning(f"Could not generate anomaly distribution: {str(e)}")
                                    else:
                                        st.info("Enable transaction tagging to see anomaly distribution by category")
                            else:
                                st.warning("No transaction data to display")
                        else:
                            st.warning("No transaction data could be extracted from this file.")
                    else:
                        st.warning("File was not processed. " + file_info.get('error', 'Unsupported file type.'))
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    if "poppler" in str(e).lower():
                        st.error("""
                        **Poppler is required for PDF processing.**
                        
                        Please install Poppler:
                        1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/
                        2. Extract to `C:\\poppler`
                        3. Add `C:\\poppler\\Library\\bin` to your system PATH
                        4. Restart the application
                        """)
    
    elif selected == "Dashboard" and st.session_state.get('file_processed', False):
        st.title("Transaction Dashboard")
       
        st.write("Transaction analysis dashboard will be displayed here.")
        
    elif selected == "Reports":
        st.title("Reports")
        st.write("Generate and download reports here.")
        
    elif selected == "Settings":
        st.title("Settings")
        st.write("Configure application settings here.")

if __name__ == "__main__":
    main()
