import streamlit as st
import pandas as pd
import plotly.express as px

def show_reports(df):
    st.title("Transaction Analytics Reports")
    
   
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    
    
    st.header("Top 5 Credits")
    credits = df[df['Type'] == 'credit']
    top_credits = credits.nlargest(5, 'Amount')
    
    if not top_credits.empty:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            total_credits = credits['Amount'].sum() if not credits.empty else 0
            avg_credit = top_credits['Amount'].mean() if not top_credits.empty else 0
            st.metric("Total Credits", f"${total_credits:,.2f}")
            st.metric("Average Credit", f"${avg_credit:,.2f}")
        
        with col2:
            fig = px.bar(
                top_credits,
                x='Amount',
                y='Description',
                orientation='h',
                title='Top 5 Credits by Amount',
                labels={'Amount': 'Amount ($)', 'Description': 'Merchant'},
                color='Amount',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No credit transactions found.")
    
    st.markdown("---")
    
    # Top 5 Debits
    st.header("Top 5 Debits")
    debits = df[df['Type'] == 'debit']
    top_debits = debits.nsmallest(5, 'Amount')
    
    if not top_debits.empty:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            total_debits = abs(debits['Amount'].sum()) if not debits.empty else 0
            avg_debit = abs(top_debits['Amount'].mean()) if not top_debits.empty else 0
            st.metric("Total Debits", f"${total_debits:,.2f}")
            st.metric("Average Debit", f"${avg_debit:,.2f}")
        
        with col2:
            
            display_debits = top_debits.copy()
            display_debits['Amount'] = display_debits['Amount'].abs()
            
            fig = px.bar(
                display_debits,
                x='Amount',
                y='Description',
                orientation='h',
                title='Top 5 Debits by Amount',
                labels={'Amount': 'Amount ($)', 'Description': 'Merchant'},
                color='Amount',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No debit transactions found.")
    
    st.markdown("---")
    
    # Top 5 Transactions by Merchant
    st.header("Top 5 Merchants by Transaction Frequency")
    
   
    df['Merchant'] = df['Description'].str.extract(r'([A-Za-z0-9\s]+)')[0].str.strip()
    
    merchant_counts = df['Merchant'].value_counts().head(5)
    
    if not merchant_counts.empty:
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.metric("Total Unique Merchants", df['Merchant'].nunique())
            st.metric("Most Frequent Merchant", merchant_counts.index[0])
        
        with col2:
            fig = px.pie(
                names=merchant_counts.index,
                values=merchant_counts.values,
                title='Top 5 Merchants by Transaction Frequency',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
            
            
            st.subheader("Merchant Breakdown")
            merchant_data = []
            for merchant in merchant_counts.index:
                merchant_transactions = df[df['Merchant'] == merchant]
                total_spent = merchant_transactions[merchant_transactions['Amount'] < 0]['Amount'].sum() * -1
                total_received = merchant_transactions[merchant_transactions['Amount'] > 0]['Amount'].sum()
                merchant_data.append({
                    'Merchant': merchant,
                    'Transaction Count': len(merchant_transactions),
                    'Total Spent': total_spent,
                    'Total Received': total_received
                })
            
            merchant_df = pd.DataFrame(merchant_data)
            st.dataframe(merchant_df, use_container_width=True)
    else:
        st.warning("No merchant data available.")

def main():
    
    if 'file_info' not in st.session_state or not st.session_state.file_info.get('processed', False):
        st.warning("Please upload and process a bank statement first from the Home page.")
        return
    
   
    df = st.session_state.file_info['data']
    
    
    if hasattr(df, 'to_native'):
        df = df.to_native()
    
    
    df.columns = [str(col).lower() for col in df.columns]
    
    
    if 'datetime' in df.columns and 'date' in df.columns:
        df = df.drop('date', axis=1)
    
    
    column_mapping = {
        'amount': 'Amount',
        'description': 'Description',
        'date': 'Date',
        'datetime': 'Date',  
        'balance': 'Balance',
        'closing_balance': 'Balance',
        'type': 'Type'
    }
    
    
    new_columns = {}
    for orig_col in df.columns:
        if orig_col in column_mapping:
            new_col = column_mapping[orig_col]
            if new_col not in new_columns:  
                new_columns[new_col] = df[orig_col]
    
    
    df = pd.DataFrame(new_columns)
    
   
    if 'Amount' in df.columns:
        if df['Amount'].dtype == 'object':
            df['Amount'] = df['Amount'].astype(str).str.replace(r'[^\d.-]', '', regex=True)
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    
    
    if 'Type' not in df.columns and 'type' in df.columns:
        df['Type'] = df['type']
    
    
    if 'Type' not in df.columns and 'Amount' in df.columns:
        df['Type'] = df['Amount'].apply(lambda x: 'credit' if x >= 0 else 'debit')
    elif 'Type' in df.columns:
        
        df['Type'] = df['Type'].str.lower().str.strip()
        
        type_mapping = {
            'credit': 'credit',
            'cr': 'credit',
            'debit': 'debit',
            'dr': 'debit',
            'withdrawal': 'debit',
            'deposit': 'credit'
        }
        df['Type'] = df['Type'].map(type_mapping).fillna(df['Type'])
    
   
    if 'Type' in df.columns and 'Amount' in df.columns:
       
        df['_amount_sign'] = df['Amount'].apply(lambda x: -1 if x < 0 else 1)
        
        df['Amount'] = df['Amount'].abs()
      
        df.loc[df['Type'] == 'debit', 'Amount'] = -df['Amount']
    
    
    if 'Date' in df.columns:
       
        df['Date'] = df['Date'].astype(str)
       
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='mixed')
       
        df = df.dropna(subset=['Date'])
    
    show_reports(df)

if __name__ == "__main__":
    main()
