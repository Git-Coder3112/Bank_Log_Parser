# 🏦 Bank Statement Analyzer & Fraud Detection System

Welcome to the  bank statement analysis ! This application is like having a personal financial assistant that can read, understand, and analyze your bank statements to help you track spending, categorize transactions, and even detect potential fraud.

## 🌟 What Does This Project Do?

Imagine you have a pile of bank statements (PDFs, Excel files, or even photos of statements). This tool can:
- 📄 Read and extract information from these documents
- 🏷️ Automatically categorize your spending (food, shopping, bills, etc.)
- 🕵️‍♂️ Detect unusual or potentially fraudulent transactions
- 📊 Show you beautiful charts and graphs of your spending habits
- 🔍 Help you find patterns in your transactions

## 🧩 Key Features

### 1. Smart Document Understanding
- **Multiple File Formats**: Works with PDFs, Excel files, and even images of bank statements
- **AI-Powered Processing**: Uses advanced AI to read and understand your bank statements
- **Fallback Systems**: If the AI isn't sure, it uses smart rules to still give you good results

### 2. Transaction Analysis
- **Automatic Categorization**: Transactions are automatically sorted into categories like Food, Shopping, Bills, etc.
- **Spending Patterns**: See how much you spend in different categories over time
- **Income vs. Expenses**: Get a clear picture of your monthly cash flow

### 3. Fraud Detection
- **Anomaly Detection**: Flags unusual transactions that don't match your normal spending
- **Risk Scoring**: Each transaction gets a risk score from 1-10
- **Explanations**: Tells you why a transaction was flagged as suspicious

### 4. User-Friendly Interface
- **Interactive Dashboard**: All your financial data in one beautiful, easy-to-use interface
- **Responsive Design**: Works on both computers and mobile devices
- **No Technical Skills Needed**: Designed to be used by anyone, anywhere

## 🛠️ How It Works (In Simple Terms)

1. **Upload Your Statement**: You give the app your bank statement
2. **Document Processing**: 
   - The app reads the text from your document
   - It identifies different parts like dates, amounts, and descriptions
3. **Data Analysis**:
   - Transactions are categorized (food, bills, etc.)
   - The system looks for unusual patterns
4. **Results Display**:
   - You see a clean dashboard with all your transactions
   - Suspicious transactions are highlighted
   - You get useful insights about your spending

## 🚀 Technology Stack

### Backend (The Brain)
- **Python**: The main programming language used
- **Streamlit**: Turns our Python code into a beautiful web app
- **Pandas**: For handling and analyzing data (like a super-powered Excel)

### Machine Learning (The Smart Part)
- **BERT**: Helps understand and categorize transaction descriptions
- **LayoutLMv3**: Special AI for reading and understanding documents
- **Isolation Forest**: Algorithm that finds unusual transactions
- **SHAP**: Explains why the AI thinks a transaction might be suspicious

### File Processing
- **PyPDF2**: Reads PDF files
- **OpenCV & PIL**: Work with images of bank statements
- **pandas**: Handles Excel and CSV files

### Data Visualization
- **Plotly**: Creates interactive charts and graphs
- **Matplotlib**: For additional charting capabilities

## 🛠️ Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step-by-Step Setup

1. **Get the Code**
   ```bash
   # Copy the project to your computer
   git clone https://github.com/yourusername/bank-log-analyzer.git
   cd bank-log-analyzer
   ```

2. **Create a Virtual Environment** (Like a Sandbox)
   ```bash
   # For Windows
   python -m venv venv
   .\venv\Scripts\activate

   # For Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   This installs all the necessary software pieces the app needs to run.

4. **Run the Application**
   ```bash
   streamlit run app/main.py
   ```

5. **Open in Browser**
   - The app will automatically open in your default web browser
   - If not, go to: http://localhost:8501

## 📂 Project Structure

```
bank_log_analyzer/
├── app/                           # Main application folder
│   ├── __init__.py               # Makes Python treat the directory as a package
│   ├── main.py                   # The heart of the application
│   ├── pages/                    # Additional pages in the app
│   │   └── dashboard.py          # The main dashboard page
│   └── utils/                    # Helper functions
│       ├── __init__.py
│       ├── file_handlers.py      # Handles different file types
│       ├── fraud_detector.py     # Detects suspicious transactions
│       ├── layoutlm_processor.py # Processes documents with AI
│       └── transaction_tagger.py # Categorizes transactions
├── requirements.txt              # List of all required Python packages
└── README.md                     # This file!
```

## 🧠 Machine Learning in Action

### 1. Transaction Categorization
- **How it works**: The app reads the description of each transaction and uses AI to decide what category it belongs to (like "Food" or "Shopping")
- **Fallback**: If the AI isn't sure, it uses simple rules (like looking for words like "restaurant" or "grocery")

### 2. Fraud Detection
- **How it works**: The system learns your normal spending patterns and flags anything unusual
- **Example**: If you usually spend ₹500 on groceries but suddenly there's a ₹50,000 transaction, it gets flagged
- **Fallback**: If the AI model fails, the system can still flag transactions based on simple rules (like very large amounts)

### 3. Document Understanding
- **How it works**: Uses LayoutLMv3 to understand the structure of bank statements
- **Fallback**: If the AI can't understand the document, it falls back to simpler text extraction methods

## 🎯 Who Is This For?

- **Individuals** who want to better understand their spending
- **Small Businesses** that need to track expenses
- **Accountants** who want to automate transaction categorization
- **Security Teams** looking to detect fraudulent transactions

## 🔍 Example Use Cases

1. **Personal Finance Management**
   - Track where your money goes each month
   - Set budgets for different categories
   - Find ways to save money

2. **Expense Reporting**
   - Automatically categorize business expenses
   - Generate reports for tax purposes
   - Track reimbursable expenses

3. **Fraud Detection**
   - Get alerts for unusual transactions
   - Review potential fraud cases
   - Keep your money safe

## 🛡️ Security & Privacy

- **Your Data Stays Yours**: All processing happens on your computer
- **No Internet Required**: Works completely offline
- **No Bank Credentials Needed**: You just upload statements manually

## 🚀 Getting Started with Development

If you want to modify or improve the app:

1. **Understand the Code**
   - Start with `app/main.py` to see how everything fits together
   - The `utils/` folder contains all the smart parts

2. **Make Changes**
   - Add new transaction categories in `transaction_tagger.py`
   - Improve fraud detection in `fraud_detector.py`
   - Add support for new file types in `file_handlers.py`

3. **Test Your Changes**
   ```bash
   # Run the tests
   python -m pytest
   
   # Or run the app to see your changes
   streamlit run app/main.py
   ```

## 🤝 Contributing

We welcome contributions! Here's how you can help:
1. Report bugs
2. Suggest new features
3. Improve documentation
4. Fix issues and submit pull requests

