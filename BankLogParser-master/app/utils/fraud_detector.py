import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union
import logging
import shap
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class FraudExplanation:
    """Class to hold fraud detection explanation results."""
    is_fraud: bool
    anomaly_score: float
    feature_importance: Dict[str, float]
    explanation_text: str


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetector:
    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        """
        Initialize the Fraud Detector with Isolation Forest.
        
        Args:
            contamination: The proportion of outliers in the data set (0-0.5)
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.explainer = None
        self.feature_importances_ = None
        self.features = [
            'amount',
            'hour',
            'day_of_week',
            'day_of_month',
            'is_weekend',
            'is_night',
            'amount_zscore',
            'amount_diff_rolling_mean',
            'transaction_frequency'
        ]
        self.is_fitted = False

    def _clean_amount(self, amount):
        """Convert amount string with currency symbols to float."""
        if pd.isna(amount):
            return 0.0
        if isinstance(amount, (int, float)):
            return float(amount)
        
        amount_str = str(amount).strip()
        amount_str = ''.join(c for c in amount_str if c.isdigit() or c in '.-')
        try:
            return float(amount_str) if amount_str else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if not hasattr(self, 'X_train_'):
            self.X_train_ = df.copy()
            
        
        df = df.copy()
        
        
        if 'amount' in df.columns:
            df['amount'] = df['amount'].apply(self._clean_amount)
        
        
        date_col = 'date' if 'date' in df.columns else 'datetime'
        if date_col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        
        date_col = 'date' if 'date' in df.columns else 'datetime'
        if date_col in df.columns and not df[date_col].isna().all():
            
            df['hour'] = df[date_col].dt.hour
            df['day_of_week'] = df[date_col].dt.dayofweek
            df['day_of_month'] = df[date_col].dt.day
            df['is_weekend'] = df[date_col].dt.dayofweek.isin([5, 6]).astype(int)
            df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        else:
            
            df['hour'] = 12
            df['day_of_week'] = 0
            df['day_of_month'] = 1
            df['is_weekend'] = 0
            df['is_night'] = 0
        
        
        df['amount_abs'] = df['amount'].abs()
        
        
        if len(df) > 1 and df['amount_abs'].std() > 0:
            df['amount_zscore'] = (df['amount_abs'] - df['amount_abs'].mean()) / df['amount_abs'].std()
        else:
            df['amount_zscore'] = 0
            
        
        if len(df) >= 7:
            df['amount_rolling_mean'] = df['amount_abs'].rolling(window=7, min_periods=1).mean()
            df['amount_rolling_std'] = df['amount_abs'].rolling(window=7, min_periods=1).std().fillna(0)
            df['amount_diff_rolling_mean'] = df['amount_abs'] - df['amount_rolling_mean']
        else:
            df['amount_rolling_mean'] = df['amount_abs'].mean()
            df['amount_rolling_std'] = 0
            df['amount_diff_rolling_mean'] = 0
            
        
        if 'date' in df.columns and len(df) > 1:
            df['date'] = pd.to_datetime(df['date'])
            df['transaction_frequency'] = df.groupby('date')['date'].transform('count')
        else:
            df['transaction_frequency'] = 1
            
        
        for feature in self.features:
            if feature not in df.columns:
                if feature == 'amount_diff_rolling_mean':
                    df[feature] = df['amount_abs'] - df.get('amount_rolling_mean', df['amount_abs'].mean())
                elif feature == 'amount_zscore' and 'amount_zscore' not in df.columns:
                    df['amount_zscore'] = 0
                elif feature not in df.columns:
                    df[feature] = 0
        
        
        df = df.drop(['amount_abs', 'amount_rolling_mean', 'amount_rolling_std'], axis=1, errors='ignore')
        
        return df

    def fit(self, df: pd.DataFrame) -> 'FraudDetector':
        """
        Fit the Isolation Forest model on transaction data.
        
        Args:
            df: DataFrame containing transaction data
            
        Returns:
            self: Returns an instance of self
        """
        # Extract features
        df_features = self._extract_features(df)
        
        # Select and scale features
        X = df_features[self.features].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the model
        self.model.fit(X_scaled)
        
        # Initialize SHAP explainer
        background = shap.utils.sample(X_scaled, 100)  
        self.explainer = shap.TreeExplainer(self.model, background, feature_names=self.features)
        
        
        shap_values = self.explainer.shap_values(X_scaled)
        self.feature_importances_ = np.abs(shap_values).mean(0)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: pd.DataFrame, return_explanations: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, List[FraudExplanation]]]:
        """
        Predict anomalies in the data with optional explanations.
        
        Args:
            X: DataFrame containing transaction data
            return_explanations: If True, returns explanations along with predictions
            
        Returns:
            If return_explanations is False, returns array of predictions (-1 for anomalies, 1 for normal).
            If return_explanations is True, returns a tuple of (predictions, explanations).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
            
        X_processed = self._extract_features(X)
        X_scaled = self.scaler.transform(X_processed[self.features])
        predictions = self.model.predict(X_scaled)
        
        if not return_explanations:
            return predictions
            
        
        explanations = []
        shap_values = self.explainer.shap_values(X_scaled)
        anomaly_scores = -self.model.score_samples(X_scaled)  
        
        for i in range(len(X_scaled)):
            
            instance_shap = shap_values[i] if isinstance(shap_values, list) else shap_values[i, :]
            
            
            feature_importance = dict(zip(self.features, instance_shap))
            
            
            explanation_text = self._generate_explanation(
                X_processed.iloc[i], 
                feature_importance,
                anomaly_scores[i],
                predictions[i] == -1
            )
            
            explanations.append(
                FraudExplanation(
                    is_fraud=(predictions[i] == -1),
                    anomaly_score=anomaly_scores[i],
                    feature_importance=feature_importance,
                    explanation_text=explanation_text
                )
            )
            
        return predictions, explanations

    def _generate_explanation(self, instance, feature_importance, anomaly_score, is_fraud):
        """
        Generate a human-readable explanation for a single instance.
        
        Args:
            instance: DataFrame row representing the instance
            feature_importance: Dictionary of feature importances
            anomaly_score: Anomaly score for the instance (0-1, higher is more suspicious)
            is_fraud: Whether the instance is predicted as fraud
            
        Returns:
            Human-readable explanation string
        """
        
        risk_level = int(anomaly_score * 10)
        risk_level = min(max(risk_level, 0), 10)  
        
        
        explanation = f"🔍 **Risk Level: {risk_level}/10**\n"
        
        
        if risk_level >= 8:
            explanation += "🚨 **High Risk Transaction**\n"
        elif risk_level >= 5:
            explanation += "⚠️ **Moderate Risk Transaction**\n"
        else:
            explanation += "✅ **Low Risk Transaction**\n"
        
       
        top_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        
        
        if top_features:
            explanation += "\n**Key Factors:**\n"
            
            feature_descriptions = {
                'amount': 'Transaction Amount',
                'hour': 'Time of Day',
                'day_of_week': 'Day of Week',
                'day_of_month': 'Day of Month',
                'is_weekend': 'Weekend Transaction',
                'is_night': 'Night-time Transaction',
                'amount_zscore': 'Unusual Amount',
                'amount_diff_rolling_mean': 'Amount vs. Typical',
                'transaction_frequency': 'Transaction Frequency'
            }
            
            for feature, importance in top_features:
                abs_importance = abs(importance)
                feature_name = feature_descriptions.get(feature, feature.replace('_', ' ').title())
                
                
                if abs_importance < 0.1:
                    continue
                    
                
                if feature == 'amount':
                    amount = abs(instance.get('amount', 0))
                    if importance > 0:
                        explanation += f"• High amount: ₹{amount:,.2f} (unusually large)\n"
                    else:
                        explanation += f"• Low amount: ₹{amount:,.2f} (unusually small)\n"
                        
                elif feature == 'hour':
                    hour = int(instance.get('hour', 12))
                    if 0 <= hour < 5:
                        time_desc = "late night"
                    elif 5 <= hour < 12:
                        time_desc = "morning"
                    elif 12 <= hour < 17:
                        time_desc = "afternoon"
                    elif 17 <= hour < 22:
                        time_desc = "evening"
                    else:
                        time_desc = "night"
                    explanation += f"• Unusual transaction time: {time_desc} ({hour}:00)\n"
                    
                elif feature == 'is_weekend' and abs_importance > 0.3:
                    explanation += "• Unusual weekend transaction pattern\n"
                    
                elif feature == 'is_night' and abs_importance > 0.3:
                    explanation += "• Unusual night-time transaction\n"
                    
                elif feature == 'amount_zscore' and abs_importance > 0.3:
                    explanation += f"• Amount significantly different from usual\n"
                    
                elif feature == 'transaction_frequency' and abs_importance > 0.3:
                    explanation += f"• Unusual transaction frequency\n"
                    
                else:
                   
                    direction = "Higher than usual" if importance > 0 else "Lower than usual"
                    explanation += f"• {feature_name}: {direction}\n"
        
       
        if is_fraud:
            explanation += "\n**Recommendation:** Review this transaction carefully. "
            explanation += "If unexpected, consider contacting your bank immediately."
        else:
            explanation += "\n**Assessment:** This transaction appears normal based on your spending patterns."
            
        return explanation

    def get_anomaly_descriptions(self, df: pd.DataFrame) -> List[Dict]:
        """
        Generate human-readable descriptions for detected anomalies.
        
        Args:
            df: DataFrame with anomaly predictions (must contain 'is_anomaly' column)
            
        Returns:
            List of dictionaries containing anomaly descriptions
        """
        anomalies = []
        
        for _, row in df[df['is_anomaly'] == 1].iterrows():
            
            date_value = row.get('date') or row.get('datetime')
            if pd.isna(date_value):
                date_str = 'Unknown date'
            else:
                try:
                    date_str = date_value.strftime('%Y-%m-%d %H:%M')
                except (AttributeError, ValueError):
                    date_str = str(date_value)
            
            description = {
                'date': date_str,
                'amount': row.get('amount', 0),
                'description': str(row.get('description', 'No description')),
                'anomaly_score': row.get('anomaly_score', 0),
                'reasons': []
            }
            
            
            if 'is_night' in row and row['is_night'] == 1:
                description['reasons'].append("Transaction occurred during night hours")
                
            if 'amount_zscore' in row and abs(row['amount_zscore']) > 3:
                description['reasons'].append(f"Unusually large amount (Z-score: {row['amount_zscore']:.2f})")
                
            if 'transaction_frequency' in row and row['transaction_frequency'] > 10:
                description['reasons'].append(f"High transaction frequency ({row['transaction_frequency']} transactions)")
                
            
            if 'amount' in row and pd.notna(row['amount']):
                amount = float(row['amount'])
                if amount > 100000:  
                    description['reasons'].append(f"Large transaction amount: ₹{amount:,.2f}")
            
            anomalies.append(description)
            
        
        anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)
        
        return anomalies
