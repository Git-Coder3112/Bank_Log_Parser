import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass

@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float
    page: int
    label: str = ""

class LayoutLMProcessor:
    def __init__(self, model_name="microsoft/layoutlmv3-base"):
        """
        Initialize the LayoutLM processor with a pre-trained model.
        
        Args:
            model_name: Name of the pre-trained model (default: microsoft/layoutlmv3-base)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = LayoutLMv3Processor.from_pretrained(model_name)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name).to(self.device)
        
        
        self.label2id = {
            "O": 0,
            "B-DATE": 1, "I-DATE": 2,
            "B-AMOUNT": 3, "I-AMOUNT": 4,
            "B-DESCRIPTION": 5, "I-DESCRIPTION": 6,
            "B-TRANSACTION_TYPE": 7, "I-TRANSACTION_TYPE": 8,
            "B-BALANCE": 9, "I-BALANCE": 10
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
    
    def process_document(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process bank statement pages using LayoutLM.
        
        Args:
            image_paths: List of paths to bank statement page images
            
        Returns:
            List of dictionaries containing extracted entities per page
        """
        all_entities = []
        
        for page_num, image_path in enumerate(image_paths):
            try:
                
                image = Image.open(image_path).convert("RGB")
                width, height = image.size
                
                
                encoding = self.processor(
                    image,
                    return_tensors="pt",
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=512,
                    padding="max_length"
                )
                
                
                inputs = {k: v.to(self.device) for k, v in encoding.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
               
                predictions = outputs.logits.argmax(-1).squeeze().tolist()
                offset_mapping = encoding.offset_mapping.squeeze().tolist()
                
                
                entities = self._extract_entities(predictions, offset_mapping, width, height, page_num)
                all_entities.extend(entities)
                
            except Exception as e:
                print(f"Error processing page {page_num + 1}: {str(e)}")
                continue
                
        return all_entities
    
    def _extract_entities(self, predictions, offset_mapping, width, height, page_num):
        """Extract and format entities from model predictions."""
        entities = []
        current_entity = None
        
        for pred, (start, end) in zip(predictions, offset_mapping):
            if start == 0 and end == 0:  
                continue
                
            label = self.id2label.get(pred, "O")
            
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "label": label[2:],  
                    "text": "",
                    "bbox": [0, 0, 0, 0],  
                    "page": page_num
                }
            elif label.startswith("I-") and current_entity:
                current_entity["text"] += " "
            
            
            
        if current_entity:
            entities.append(current_entity)
            
        return entities
    
    def extract_transactions(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert extracted entities into structured transaction data.
        
        Args:
            entities: List of entities from process_document
            
        Returns:
            List of structured transaction dictionaries
        """
        transactions = []
        current_transaction = {}
        
        for entity in entities:
            label = entity["label"]
            text = entity["text"].strip()
            
            if label == "DATE":
                if current_transaction:  
                    transactions.append(current_transaction)
                current_transaction = {"date": text}
            elif label == "DESCRIPTION" and current_transaction:
                current_transaction["description"] = text
            elif label == "AMOUNT" and current_transaction:
                try:
                    amount = float(text.replace(",", ""))
                    current_transaction["amount"] = amount
                except (ValueError, AttributeError):
                    current_transaction["amount"] = text
            elif label == "TRANSACTION_TYPE" and current_transaction:
                current_transaction["type"] = text
            elif label == "BALANCE" and current_transaction:
                try:
                    balance = float(text.replace(",", ""))
                    current_transaction["balance"] = balance
                except (ValueError, AttributeError):
                    current_transaction["balance"] = text
        
        
        if current_transaction:
            transactions.append(current_transaction)
            
        return transactions
