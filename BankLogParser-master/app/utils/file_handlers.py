import pandas as pd
from io import BytesIO
import os
import tempfile
from pdf2image import convert_from_bytes
from .pdf_processor import PDFProcessor
from .layoutlm_processor import LayoutLMProcessor

class FileHandler:
    def __init__(self, use_ml=True):
        self.pdf_processor = PDFProcessor()
        self.use_ml = use_ml
        self.ml_processor = LayoutLMProcessor() if use_ml else None
    
    def handle_file_upload(self, uploaded_file):
        """Handle file upload and return processed data."""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            return self._process_pdf(uploaded_file)
        else:
            return self._get_file_info(uploaded_file)
    
    def _process_pdf(self, uploaded_file):
        """Process PDF file and extract structured data using either ML or rule-based approach."""
        
        file_info = self._get_file_info(uploaded_file)
        
        try:
            
            pdf_bytes = uploaded_file.getvalue()
            
            
            if not pdf_bytes:
                raise ValueError("Uploaded file is empty")
            
            df = None
            
            
            if self.use_ml and self.ml_processor:
                try:
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        
                        images = convert_from_bytes(pdf_bytes)
                        image_paths = []
                        
                        
                        for i, image in enumerate(images):
                            image_path = os.path.join(temp_dir, f'page_{i+1}.jpg')
                            image.save(image_path, 'JPEG')
                            image_paths.append(image_path)
                        
                        
                        entities = self.ml_processor.process_document(image_paths)
                        transactions = self.ml_processor.extract_transactions(entities)
                        
                       
                        if transactions:
                            df = pd.DataFrame(transactions)
                            if hasattr(df, 'to_native'):
                                df = df.to_native()
                except Exception as e:
                    print(f"ML-based processing failed, falling back to rule-based: {str(e)}")
            
            
            if df is None or df.empty:
                df = self.pdf_processor.process_pdf(pdf_bytes)
            
            if df is None or df.empty:
                raise ValueError("No transaction data could be extracted from the PDF")
                
            
            if hasattr(df, 'to_native'):
                df = df.to_native()
            
            
            expected_columns = {
                'date': '',
                'time': '',
                'type': '',
                'description': '',
                'amount': 0.0,
                'location': '',
                'closing_balance': 0.0
            }
            
            for col, default_val in expected_columns.items():
                if col not in df.columns:
                    df[col] = default_val
            
           
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                except:
                    pass
            
            
            if 'date' in df.columns and not df['date'].isna().all():
                df = df.sort_values('date')
            
           
            file_info['data'] = df
            file_info['processed'] = True
            
            return file_info
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            file_info['error'] = error_msg
            file_info['processed'] = False
            return file_info
    
    def _get_file_info(self, uploaded_file):
        """Get basic file information."""
        file_extension = uploaded_file.name.split('.')[-1].lower()
        return {
            'filename': uploaded_file.name,
            'type': file_extension.upper(),
            'size': len(uploaded_file.getvalue()),
            'content': uploaded_file.getvalue(),
            'processed': False,
            'data': None,
            'error': None
        }


__all__ = ['FileHandler']
