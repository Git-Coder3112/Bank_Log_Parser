import re
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 10
MAX_LENGTH = 128 

class TransactionTagger:
    def __init__(self, use_gpu=False):
        """
        Initialize the transaction tagger with BERT and rule-based patterns.
        
        Args:
            use_gpu: Whether to use GPU for BERT inference if available
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        self.tokenizer = None
        self.model = None
        self.nlp = None
        self._load_models()
        
        
        self.label_map = {
            0: 'food',
            1: 'shopping',
            2: 'transport',
            3: 'bills',
            4: 'groceries',
            5: 'salary',
            6: 'transfer',
            7: 'investment',
            8: 'travel',
            9: 'health'
        }
        
        self.categories = {
            'food': {
                'patterns': [
                    r'(?i)(?:swiggy|zomato|restaurant|cafe|mcdonalds|kfc|pizza|burger|food|dine|eat|meal|coffee)',
                    r'(?i)(?:starbucks|barista|dominos|subway|foodpanda|ubereats|deliveroo|grubhub)'
                ],
                'amount_ranges': [(0, 5000)],  
                'description_keywords': ['food', 'lunch', 'dinner', 'breakfast', 'cafe', 'restaurant']
            },
            'shopping': {
                'patterns': [
                    r'(?i)(?:amazon|flipkart|myntra|ajio|shop|store|retail|purchase|buy|big\s?bazaar|bigbazaar)',
                    r'(?i)(?:fashion|clothing|apparel|electronics|gadget|device|supermarket|mart)'
                ],
                'amount_ranges': [(100, 100000)],
                'description_keywords': ['shop', 'store', 'purchase', 'order', 'bigbazaar', 'big bazaar', 'retail']
            },
            'transport': {
                'patterns': [
                    r'(?i)(?:uber|ola|rapido|lyft|taxi|cab|auto|rickshaw|metro|bus|train|flight|airport|irctc|railway|rail)',
                    r'(?i)(?:petrol|diesel|fuel|gas|transport|travel|ticket|booking|journey|passenger)',
                    r'(?i)(?:irctc.*ticket|train.*ticket|flight.*ticket|bus.*ticket)'
                ],
                'amount_ranges': [(10, 50000)],  
                'description_keywords': ['cab', 'taxi', 'uber', 'ola', 'fuel', 'ticket', 'irctc', 'railway', 'train', 'flight', 'bus', 'booking', 'passenger']
            },
            'bills': {
                'patterns': [
                    r'(?i)(?:bill|payment|postpaid|prepaid|jio|airtel|vi|vodafone|idea|paytm)',
                    r'(?i)(?:electricity|water|gas|internet|broadband|wifi|mobile|phone|recharge|dth)',
                    r'(?i)(?:paytm\s?(?:money|recharge|payment|wallet)?)'
                ],
                'amount_ranges': [(10, 100000)],
                'description_keywords': ['bill', 'payment', 'recharge', 'topup', 'paytm', 'mobile recharge', 'dth']
            },
            'groceries': {
                'patterns': [
                    r'(?i)(?:bigbasket|grofers|dmart|reliance|more|supermarket|grocery|mart|big\s?bazaar|bigbazaar)',
                    r'(?i)(?:vegetable|fruit|milk|bread|egg|rice|wheat|pulse|dal)'
                ],
                'amount_ranges': [(100, 20000)],
                'description_keywords': ['grocery', 'mart', 'supermarket', 'provision', 'bigbazaar', 'big bazaar']
            },
            'entertainment': {
                'patterns': [
                    r'(?i)(?:netflix|prime|hotstar|disney|movie|cinema|theater|concert|show|event|bookmyshow)',
                    r'(?i)(?:bms|book my show|movie tickets?|event tickets?)'
                ],
                'amount_ranges': [(100, 10000)],
                'description_keywords': ['movie', 'show', 'ticket', 'entertainment', 'bookmyshow', 'bms', 'event']
            },
            'salary': {
                'patterns': [
                    r'(?i)(?:salary|income|credit|payroll|stipend|pension|allowance)',
                    r'(?i)(?:credit.*salary|salary.*credit|salary.*credit)'
                ],
                'amount_ranges': [(10000, 1000000)],
                'description_keywords': ['salary', 'credit', 'income', 'payroll']
            },
            'transfer': {
                'patterns': [
                    r'(?i)(?:transfer|upi|imps|neft|rtgs|nfs|atm|withdrawal|deposit|bank)',
                    r'(?i)(?:sent to|received from|to account|from account)'
                ],
                'amount_ranges': [(10, 10000000)],
                'description_keywords': ['transfer', 'upi', 'imps', 'neft', 'rtgs']
            },
            'investment': {
                'patterns': [
                    r'(?i)(?:mutual|sip|stock|equity|fd|fixed deposit|recurring deposit|rd|nps|ppf)',
                    r'(?i)(?:investment|savings|portfolio|trading|demat|brokerage)'
                ],
                'amount_ranges': [(1000, 1000000)],
                'description_keywords': ['investment', 'sip', 'fd', 'mutual', 'stock']
            },
            'other': {
                'patterns': ['.*'],  
                'amount_ranges': [],
                'description_keywords': []
            }
        }
        
        
        self.rules = {
            'food': [r'zomato', r'swiggy', r'restaurant', r'eat', r'cafe', r'dine', r'food'],
            'shopping': [r'amazon', r'flipkart', r'myntra', r'shop', r'store', r'retail'],
            'transport': [r'uber', r'ola', r'rapido', r'cab', r'taxi', r'metro', r'bus'],
            'bills': [r'bill', r'payment', r'postpaid', r'prepaid', r'jio', r'airtel', r'vi'],
            'groceries': [r'bigbasket', r'grofers', r'dmart', r'relief', r'grocery', r'supermarket'],
            'travel': [r'booking', r'mmt', r'makemytrip', r'goibibo', r'flight', r'train', r'hotel'],
            'health': [r'pharmacy', r'med', r'hospital', r'clinic', r'doc', r'health'],
            'salary': [r'salary', r'income', r'credit', r'payroll'],
            'investment': [r'mutual', r'sip', r'stock', r'mf', r'fd', r'fixed deposit'],
            'transfer': [r'transfer', r'upi', r'imps', r'neft', r'rtgs']
        }
    
    def _load_models(self):
        """Load the BERT model and tokenizer."""
        try:
            logger.info("Loading BERT model and tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME,
                num_labels=NUM_LABELS
            )
            
           
            self.nlp = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if str(self.device) == 'cuda' else -1
            )
            logger.info("BERT model and tokenizer loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading BERT model: {str(e)}")
            logger.warning("Falling back to rule-based tagging only")
            self.nlp = None
    
    def _predict_with_bert(self, text: str) -> Tuple[str, float]:
        """
        Predict category using BERT model.
        
        Args:
            text: Transaction description
            
        Returns:
            Tuple of (predicted_category, confidence_score)
        """
        if not self.nlp or not text or not isinstance(text, str):
            return 'uncategorized', 0.0
            
        try:
            
            text = text.lower().strip()
            if not text:
                return 'uncategorized', 0.0
                
            
            result = self.nlp(text, truncation=True, max_length=MAX_LENGTH)
            
            
            if isinstance(result, list) and result:
                pred = result[0]
                label_idx = int(pred['label'].split('_')[-1])
                confidence = pred['score']
                
                
                category = self.label_map.get(label_idx, 'uncategorized')
                return category, float(confidence)
                
        except Exception as e:
            logger.error(f"Error in BERT prediction: {str(e)}")
            
        return 'uncategorized', 0.0
    
    def _matches_pattern(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the given patterns."""
        if not text or not isinstance(text, str):
            return False
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def _is_amount_in_range(self, amount: float, ranges: List[tuple]) -> bool:
        """Check if amount falls within any of the specified ranges."""
        if not ranges:  
            return True
            
        try:
            amount = float(amount)
            return any(lower <= amount <= upper for lower, upper in ranges)
        except (ValueError, TypeError):
            return False
    
    def _get_keyword_matches(self, text: str, keywords: List[str]) -> int:
        """Count how many keywords are present in the text."""
        if not text or not isinstance(text, str):
            return 0
        text = text.lower()
        return sum(1 for kw in keywords if kw.lower() in text)
    
    def predict_category(self, description: str, amount: float) -> Tuple[str, float]:
        """
        Predict the category of a transaction using BERT with rule-based fallback.
        
        Args:
            description: Transaction description
            amount: Transaction amount
            
        Returns:
            Tuple of (predicted_category, confidence_score)
        """
        if not description or not isinstance(description, str):
            return 'uncategorized', 0.0
            
        description = str(description).strip()
        
        
        bert_category, bert_confidence = self._predict_with_bert(description)
        
        
        best_rule_category = 'uncategorized'
        best_rule_score = 0.0
        
        for category, rules in self.categories.items():
            if category == 'other':
                continue
                
            pattern_match = self._matches_pattern(description, rules['patterns'])
            amount_match = self._is_amount_in_range(amount, rules['amount_ranges'])
            keyword_matches = self._get_keyword_matches(description, rules['description_keywords'])
            
            
            score = 0.0
            if pattern_match:
                score += 0.7  
            if amount_match and rules['amount_ranges']:
                score += 0.2  
            score += min(0.1, keyword_matches * 0.05)  
            
            if score > best_rule_score:
                best_rule_score = score
                best_rule_category = category
        
        if bert_confidence >= 0.6 and bert_category != 'uncategorized':
            
            if best_rule_score >= 0.8:
                return best_rule_category, best_rule_score
            return bert_category, bert_confidence
            
        
        if best_rule_score >= 0.5:
            return best_rule_category, best_rule_score
            
       
        if bert_confidence > 0 and bert_category != 'uncategorized':
            return bert_category, bert_confidence * 0.8  
            
        
        if best_rule_score > 0:
            return best_rule_category, best_rule_score
            
        
        if description.strip() and amount != 0:
            return 'other', 0.3
            
        return 'uncategorized', 0.0
    
    def tag_transaction(self, description: str, amount: float, use_ml: bool = False) -> Dict[str, Any]:
        """
        Tag a transaction with category and confidence.
        
        Args:
            description: Transaction description
            amount: Transaction amount
            use_ml: Whether to use ML (kept for backward compatibility, not used)
            
        Returns:
            Dictionary with tag information
        """
        try:
            
            if isinstance(amount, str):
                
                amount = float(''.join(c for c in amount if c.isdigit() or c in '.-'))
            amount = float(amount) if amount is not None else 0.0
            
           
            category, confidence = self.predict_category(description or '', amount)
            
            return {
                'tag': category,
                'confidence': confidence,
                'method': 'rules',
                'amount': amount
            }
            
        except Exception as e:
            logger.error(f"Error tagging transaction: {str(e)}")
            return {
                'tag': 'error',
                'confidence': 0.0,
                'method': 'error',
                'amount': amount if isinstance(amount, (int, float)) else 0.0
            }
    
    def tag_dataframe(self, df: pd.DataFrame, description_col: str = 'description', 
                     amount_col: str = 'amount', use_ml: bool = True) -> pd.DataFrame:
        """
        Tag all transactions in a DataFrame using BERT with rule-based fallback.
        
        Args:
            df: Input DataFrame with transaction data
            description_col: Name of the description column
            amount_col: Name of the amount column
            use_ml: Whether to use BERT for tagging (with rule-based fallback)
            
        Returns:
            DataFrame with added tag columns (tag, tag_confidence, tag_method)
        """
        if df.empty:
            return df
            
        result = df.copy()
        
        
        result['tag'] = 'uncategorized'
        result['tag_confidence'] = 0.0
        result['tag_method'] = 'none'
        
        
        for idx, row in df.iterrows():
            tag_info = self.tag_transaction(
                row.get(description_col, ''),
                row.get(amount_col, 0.0),
                use_ml=use_ml
            )
            
            result.at[idx, 'tag'] = tag_info['tag']
            result.at[idx, 'tag_confidence'] = tag_info['confidence']
            result.at[idx, 'tag_method'] = tag_info['method']
        
        return result
