# Legal AI Assistant

A multilingual Legal AI Assistant built with a fine-tuned Qwen LLM, FastAPI backend, and Streamlit frontend.

## Architecture

- **Model**: Qwen2.5-7B-Instruct (4-bit) fine-tuned with LoRA
- **Backend**: FastAPI (hosted on Kaggle, exposed via ngrok)
- **Frontend**: Streamlit (local UI)
- **Output**: Structured legal answers (JSON → formatted text)
- **Languages**: English & Arabic

## Features

- Multilingual legal reasoning (EN / AR)
- Structured output (Conclusion, Related Laws, Risk Level)
- Robust JSON extraction from LLM output
- Backend-driven formatting (frontend is a dumb renderer)
- Secure API with token authentication
- LoRA fine-tuning for efficiency

## Demo

![Demo](assets/demo.mp4)

## 🚀 Running the Project
first insatll requirements:
```bash
pip install -r requirements.txt
```

### Backend 
1. Add these variables to a .end file:
   - `API_KEY`
   - `NGROK_TOKEN`
2. Run:
   ```bash
   python scripts/run_api_ngrok.py
  
### Frontend
Run:
   ```bash
   streamlit run ui/app.py
   ``` 

