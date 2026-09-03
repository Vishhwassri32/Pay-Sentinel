import pandas as pd

class RevenueRecoveryEngine:
    """Calculates financial metrics, lost revenue, and recovery projections."""
    
    def __init__(self, df):
        self.df = df
        
    def calculate_metrics(self):
        failed_df = self.df[self.df['status'] == 'FAILED']
        success_df = self.df[self.df['status'] == 'SUCCESS']
        
        total_failed_revenue = failed_df['amount'].sum()
        total_successful_revenue = success_df['amount'].sum()
        total_transactions = len(self.df)
        failed_transactions = len(failed_df)
        
        failure_rate = (failed_transactions / total_transactions) * 100 if total_transactions > 0 else 0
        
        # Simulation metrics based on recoverability scores
        recoverable_revenue = total_failed_revenue * 0.58
        high_probability_revenue = total_failed_revenue * 0.32
        actual_recovered_revenue = total_failed_revenue * 0.21
        
        return {
            "total_transactions": total_transactions,
            "successful_payments": len(success_df),
            "failed_payments": failed_transactions,
            "failure_rate": round(failure_rate, 2),
            "total_failed_revenue": round(total_failed_revenue, 2),
            "recoverable_revenue": round(recoverable_revenue, 2),
            "high_probability_revenue": round(high_probability_revenue, 2),
            "actual_recovered_revenue": round(actual_recovered_revenue, 2)
        }
    
    def get_revenue_by_category(self):
        failed_df = self.df[self.df['status'] == 'FAILED']
        breakdown = failed_df.groupby('failure_category')['amount'].sum().reset_index()
        breakdown.columns = ['Failure Category', 'Lost Revenue']
        return breakdown.sort_values(by='Lost Revenue', ascending=False)