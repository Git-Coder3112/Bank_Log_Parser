import streamlit as st
import pandas as pd
import plotly.express as px

def show_dashboard(df):
    st.title("Transaction Analysis Dashboard")
    
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", len(df))
    with col2:
        st.metric("Total Debits", f"${df[df['Amount'] < 0]['Amount'].sum():,.2f}")
    with col3:
        st.metric("Total Credits", f"${df[df['Amount'] > 0]['Amount'].sum():,.2f}")
    
    # Transaction over time
    st.subheader("Transaction History")
    df['Date'] = pd.to_datetime(df['Date'])
    df_sorted = df.sort_values('Date')
    
    # Line chart of balance over time
    fig = px.line(df_sorted, x='Date', y='Balance', 
                 title='Account Balance Over Time',
                 labels={'Balance': 'Balance ($)', 'Date': 'Date'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Top expenses
    st.subheader("Top Expenses")
    expenses = df[df['Amount'] < 0].nlargest(10, 'Amount', 'all')
    expenses['Amount'] = expenses['Amount'].abs()
    fig = px.bar(expenses, x='Amount', y='Description', 
                orientation='h',
                title='Largest Expenses',
                labels={'Amount': 'Amount ($)', 'Description': 'Description'})
    st.plotly_chart(fig, use_container_width=True)
    
    
    st.subheader("Spending by Category")
    
    st.info("Category analysis will be available after implementing transaction categorization.")

def main():
    if 'df' not in st.session_state:
        st.warning("Please upload and process a bank statement first.")
        return
    
    df = st.session_state['df']
    show_dashboard(df)

if __name__ == "__main__":
    main()
