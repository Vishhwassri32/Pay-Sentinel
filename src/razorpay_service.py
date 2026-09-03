import os
import requests
from dotenv import load_dotenv

load_dotenv()

class RazorpayService:
    """Service layer for integrating Razorpay APIs with secure key management."""
    
    def __init__(self):
        self.key_id = os.getenv("RAZORPAY_KEY_ID", "rzp_test_mock")
        self.key_secret = os.getenv("RAZORPAY_KEY_SECRET", "secret_mock")
        self.base_url = "https://api.razorpay.com/v1"
        self.is_mock = "mock" in self.key_id or not self.key_id
        
    def fetch_payments(self, count=50):
        """Fetches payment logs from Razorpay API or returns mock sandbox data."""
        if self.is_mock:
            return {"status": "success", "mode": "sandbox_mock", "count": count}
            
        try:
            response = requests.get(
                f"{self.base_url}/payments",
                auth=(self.key_id, self.key_secret),
                params={"count": count}
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"status": "error", "message": response.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}