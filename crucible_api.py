import os
from dotenv import load_dotenv
import requests
load_dotenv()

BASE_URL = 'https://vllm.i.staging-crucible.dreadnode.io/v1'
API_KEY = os.getenv('CRUCIBLE_API_KEY')

class CrucibleClient:
    def __init__(self, base_url=BASE_URL, api_key=API_KEY):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {"Authorization": api_key}
    
    def create_submission(self, file_path):
        with open(file_path, "rb") as file:
            files = {"file": (file_path, file, "application/json")}
            response = requests.post(f"{self.base_url}/submission", files=files, headers=self.headers)
            return response.json()["submission_id"]

    def get_submission(self, submission_id):
        url = f"{self.base_url}/submission/{submission_id}"
        response = requests.get(url, headers=self.headers)
        return response.json()
    
    def add_run_to_submission(self, submission_id, file_path):
        url = f"{self.base_url}/submission/{submission_id}/runs"
        with open(file_path, "rb") as file:
            files = {"file": (file_path, file, "application/json")}
            response = requests.put(url, files=files, headers=self.headers)
            return response.json()
        
    def delete_run(self, submission_id, run_id):
        url = f"{self.base_url}/submission/{submission_id}/runs/{run_id}"
        response = requests.delete(url, headers=self.headers)
        if response.status_code != 204:
            raise Exception(response.json())
        
    def upload_evidence(self, submission_id, file_path):
        url = f"{self.base_url}/submission/{submission_id}/evidence"
        with open(file_path, "rb") as file:
            files = {"file": (file_path, file, "text/plain")}
            response = requests.put(url, files=files, headers=self.headers)
            return response.json()["evidence_id"]
        
    def delete_evidence(self, submission_id, evidence_id):
        url = f"{self.base_url}/submission/{submission_id}/evidence/{evidence_id}"
        response = requests.delete(url, headers=self.headers)
        if response.status_code != 204:
            raise Exception(response.json())
        
    def delete_submission(self, submission_id):
        url = f"{self.base_url}/submission/{submission_id}"
        response = requests.delete(url, headers=self.headers)
        if response.status_code != 204:
            raise Exception(response.json())
        
