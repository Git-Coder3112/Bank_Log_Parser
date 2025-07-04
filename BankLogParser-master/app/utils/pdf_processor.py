import re
import PyPDF2
from typing import List, Dict, Tuple, Optional
import pandas as pd
from io import BytesIO
from dataclasses import dataclass

@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float
    page: int

@dataclass
class Token:
    text: str
    bbox: BoundingBox
    page_num: int

class PDFProcessor:
    def __init__(self):
        
        self.date_pattern = r'\d{1,2}/\d{1,2}/\d{2,4}'
        self.amount_pattern = r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?'

    def extract_text(self, pdf_bytes: bytes) -> List[str]:
        """Extract text from PDF using PyPDF2."""
        text_pages = []

        try:
            
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))

            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_pages.append(text)

            return text_pages

        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
            return []

    def parse_transactions(self, text_pages: List[str]) -> List[dict]:
        """Parse transaction data from text pages.
        
        Handles multi-line transactions where the first line contains the main transaction data
        and subsequent lines contain additional information like location and closing balance.
        """
        transactions = []
        current_transaction = {}
        current_lines = []

        def process_current_transaction():
            nonlocal current_transaction, current_lines
            if current_lines:
                
                full_text = ' '.join(current_lines)
                current_transaction = self._parse_transaction_line(full_text)
                if current_transaction:
                    transactions.append(current_transaction)
                current_lines = []

        for page_text in text_pages:
            for line in page_text.split('\n'):
                line = line.strip()
                if not line:
                    continue

                
                if re.match(r'^[A-F0-9]{8}\s', line):
                    process_current_transaction()  
                    current_lines = [line]  
                else:
                    current_lines.append(line)

        
        process_current_transaction()

        return transactions

    def _parse_transaction_line(self, line_text: str) -> dict:
        """Parse a single transaction line in the format:
        <transaction_id> <account> <datetime> <type> <amount> <description> [location] [method] Closing Balance: <amount>
        
        Example: 
        F59493CA ACC1012 2024-05-26 18:08:44 debit 43334.48 Uber mobile_app Mumbai branch Closing Balance: ₹131689.05
        """
        print(f"Parsing transaction line: {line_text}")

        
        transaction = {
            'transaction_id': '',
            'account': '',
            'datetime': '',  
            'type': '',
            'amount': '',
            'description': '',
            'location': '',
            'method': '',
            'closing_balance': ''
        }

        try:
            
            closing_balance_match = re.search(
                r'Closing Balance:\s*[₹$€£]?\s*([\d,]+(?:\.\d{2})?)', 
                line_text
            )
            if closing_balance_match:
                transaction['closing_balance'] = closing_balance_match.group(1).replace(',', '')
                
                line_text = line_text[:closing_balance_match.start()].strip()

            
            parts = [p for p in re.split(r'\s+', line_text) if p]
            
            if len(parts) < 6:  
                return transaction

            
            transaction['transaction_id'] = parts[0]
            
           
            if len(parts) > 1 and parts[1].startswith(('ACC', 'AC', 'A/C')):
                transaction['account'] = parts[1]
                parts = parts[2:]
            else:
                parts = parts[1:]

            
            if parts:
               
                datetime_match = re.search(
                    r'(\d{4})\s*-?\s*(\d{2})\s*-?\s*(\d{2})\s+(\d{1,2}:\d{2}(?::\d{2})?)',
                    ' '.join(parts[:3])  
                )
                if datetime_match:
                    year, month, day, time_str = datetime_match.groups()
                    transaction['datetime'] = f"{year}-{month}-{day} {time_str}"
                    
                    matched_str = datetime_match.group(0)
                    parts = ' '.join(parts).replace(matched_str, '', 1).split()
            
            
            if parts and parts[0].lower() in ['credit', 'debit']:
                transaction['type'] = parts[0].lower()
                parts = parts[1:]
            
            
            if parts:
                for i, part in enumerate(parts):
                    amount_str = part.replace(',', '').replace('₹', '').replace('$', '').replace('€', '').strip()
                    if re.match(r'^\d+\.\d{2}$', amount_str):
                        transaction['amount'] = amount_str
                        parts = parts[i+1:]
                        break
            
            
            if parts:
                desc_parts = []
                location_found = False
                method_found = False
                
               
                locations = ['Bengaluru', 'Ahmedabad', 'Kolkata', 'Lucknow', 'Mumbai', 'Unknown']
                methods = ['AUTO', 'NEFT', 'UPI', 'ATM', 'batch', 'web', 'mobile_app', 'API', 'branch']
                
                for part in parts:
                    part_lower = part.lower()
                    
                    if not method_found and any(m.lower() == part_lower for m in methods):
                        transaction['method'] = part
                        method_found = True
                    
                    elif not location_found and any(loc.lower() == part_lower for loc in locations):
                        transaction['location'] = part
                        location_found = True
                    else:
                        desc_parts.append(part)
                
                transaction['description'] = ' '.join(desc_parts).strip()
            
            print(f"Parsed transaction: {transaction}")
            
        except Exception as e:
            print(f"Error parsing transaction line: {str(e)}")

        return transaction

    def _parse_additional_info(self, line_text: str, transaction: dict) -> dict:
        """This method is no longer needed as we now handle everything in _parse_transaction_line"""
        return transaction

    def process_pdf(self, pdf_bytes: bytes) -> pd.DataFrame:
        """Process PDF and extract transaction data."""
        try:
            print("Starting PDF processing...")

            
            print("Extracting text from PDF...")
            text_pages = self.extract_text(pdf_bytes)

            if not text_pages:
                print("No text extracted from PDF")
                return pd.DataFrame()

            
            print("Parsing transactions...")
            transactions = self.parse_transactions(text_pages)

            if not transactions:
                print("No transactions found in the document")
                return pd.DataFrame()

            
            df = pd.DataFrame(transactions)

            
            if 'amount' in df.columns:
                df['amount'] = pd.to_numeric(df['amount'].str.replace('[^0-9.-]', '', regex=True), errors='coerce')

            if 'closing_balance' in df.columns:
                df['closing_balance'] = pd.to_numeric(df['closing_balance'].str.replace('[^0-9.-]', '', regex=True), errors='coerce')

            print(f"Successfully extracted {len(df)} transactions")
            print("Columns:", df.columns.tolist())

            return df

        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()