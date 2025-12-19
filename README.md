🌍 Travel Buddy
A full‑stack Travel Buddy Matching & Chat Application that helps users find like‑minded travel partners based on interests and preferences, with real‑time chat support.

✨ Project Overview
Travel Buddy allows users to:

Create a travel profile
Get intelligent travel‑buddy recommendations
Chat with matched users in real time
The project combines frontend (HTML/CSS/JS) with a FastAPI backend, uses Machine Learning (TF‑IDF + Cosine Similarity) for matching, and SQLite for chat persistence.

🚀 Features
👤 User Travel Profiles
🧠 ML‑based Buddy Matching (TF‑IDF + Cosine Similarity)
💬 Real‑time Chat using WebSockets
📂 Profile storage using CSV
🗄 Chat history stored in SQLite
🌐 CORS‑enabled API (frontend‑ready)
🛠 Tech Stack
Frontend
HTML
CSS
JavaScript
Backend
Python
FastAPI
WebSockets
SQLite
Pandas
Scikit‑learn
📁 Project Structure
Travel Buddy/
│
├── index.html          # Frontend UI
│
├── backend/
│   ├── main.py         # FastAPI backend + ML logic
│   ├── profiles.csv    # User travel profiles
│   ├── travel_chat.db  # SQLite chat database
│   └── __pycache__/
│
└── .vscode/
🧠 Matching Logic
User interests are converted into vectors using TF‑IDF Vectorizer
Cosine Similarity is applied to find the most compatible travel buddies
Results are ranked and returned via API
💬 Chat System
Uses WebSockets for real‑time communication
Messages are stored with timestamps
Supports multi‑user chat sessions
📥 Clone the Repository
To get a local copy of the project, clone the repository using Git:

git clone https://github.com/r20j/travel-buddy.git
Navigate into the project folder:

cd travel-buddy
▶️ How to Run the Project
Backend Setup
cd backend
pip install fastapi uvicorn pandas scikit-learn
uvicorn main:app --reload
Backend will run at:

http://127.0.0.1:8000
Frontend
Simply open index.html in a browser.

🔮 Future Enhancements
🔐 User authentication (JWT)
📱 Mobile‑friendly UI
🌍 Location‑based matching
☁️ Cloud database integration
🧠 Advanced recommendation models
📌 Use Case
Perfect for:

College mini‑projects
Full‑stack demos
AI‑based recommendation systems
FastAPI + ML learning projects
📝 License
This project is open‑source and free to use for educational purposes.
