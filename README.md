# AI Smart Agriculture Advisor 🌱

An intelligent, interactive agriculture assistant designed to answer complex agricultural queries. This application leverages an agentic architecture, combining Machine Learning predictions, Retrieval-Augmented Generation (RAG), and rule-based analysis to provide actionable farming insights.

## Overview

The AI Smart Agriculture Advisor uses a LangChain ReAct agent to intelligently route user questions to the most appropriate backend tool. Whether evaluating soil health, predicting the best crops for specific weather conditions, or retrieving agricultural knowledge from customized databases, the system handles it seamlessly through a clean Streamlit interface.

## Key Features

* **Intelligent Reasoning (Agent):** Utilizes a LangChain ReAct agent powered by OpenAI to understand context and select the right tool for the job.
* **Machine Learning Crop Predictor:** Employs a trained Random Forest classifier (`scikit-learn`) to recommend crops based on N-P-K values, temperature, humidity, pH, and rainfall.
* **Knowledge Retrieval (RAG):** Uses `sentence-transformers` and `FAISS` to search a custom vector database built from crop datasets and external agricultural documents.
* **Rule-Based Soil Analysis:** Rapidly evaluates soil pH levels to recommend targeted amendments (e.g., lime or compost).
* **Interactive Web UI:** Built with Streamlit for a fast, responsive, and intuitive user experience.

## Project Structure

```text
.
├── agents/
│   └── ai_agent.py             # LangChain ReAct Agent configuration
├── data/
│   └── Crop_recommendation.csv # Source dataset for ML and embeddings
├── documents/                  # Directory for supplementary .txt files
├── tools/
│   ├── crop_tool.py            # ML-based prediction tool
│   ├── rag_tool.py             # FAISS-based retrieval tool
│   └── soil_tool.py            # Rule-based pH analysis tool
├── create_vector_db.py         # Script to generate FAISS index
├── train_model.py              # Script to train the Random Forest model
├── streamlit_app.py            # Streamlit frontend application
└── README.md
```

## Prerequisites

* Python 3.8+
* OpenAI API Key

## Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/yourusername/ai-smart-agriculture-advisor.git](https://github.com/yourusername/ai-smart-agriculture-advisor.git)
cd ai-smart-agriculture-advisor
```

**2. Create and activate a virtual environment**
```bash
# macOS/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install streamlit langchain langchain-openai langchain-community scikit-learn pandas numpy sentence-transformers faiss-cpu
```

**4. Configure Environment Variables**
Create a `.env` file in the root directory and securely add your OpenAI API key:
```text
OPENAI_API_KEY=your_actual_api_key_here
```

**5. Initialize Models and Databases**
Generate the necessary `.pkl` and `.faiss` files before running the application.

*Train the Crop Prediction Model:*
```bash
python train_model.py
```

*Build the Vector Database:*
```bash
python create_vector_db.py
```

## Usage

Start the Streamlit server:
```bash
streamlit run streamlit_app.py
```

Navigate to `http://localhost:8501` in your browser. Enter soil parameters, weather conditions, or general agricultural questions into the text area to receive AI-driven advice.
```

