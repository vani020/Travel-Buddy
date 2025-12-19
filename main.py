from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import asyncio
from datetime import datetime
from typing import Dict, List
import sqlite3
import os
import re
from datetime import datetime, timedelta
import random

app = FastAPI(title="Travel Buddy ML API", debug=True)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
def init_db():
    conn = sqlite3.connect('travel_chat.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_id TEXT NOT NULL,
            receiver_id TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

class Profile(BaseModel):
    destination: str
    travel_style: str
    hobbies: str
    filter_type: str = "all"
    travel_date: str = None  # Add travel date

class ChatMessage(BaseModel):
    message: str
    user_id: str
    receiver_id: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]

    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

manager = ConnectionManager()

# Database functions
def save_message(sender_id: str, receiver_id: str, message: str):
    conn = sqlite3.connect('travel_chat.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO chat_messages (sender_id, receiver_id, message) VALUES (?, ?, ?)', 
                   (sender_id, receiver_id, message))
    conn.commit()
    conn.close()

def get_chat_history(user1_id: str, user2_id: str):
    conn = sqlite3.connect('travel_chat.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT sender_id, receiver_id, message, timestamp 
        FROM chat_messages 
        WHERE (sender_id = ? AND receiver_id = ?) OR (sender_id = ? AND receiver_id = ?)
        ORDER BY timestamp ASC
    ''', (user1_id, user2_id, user2_id, user1_id))
    messages = cursor.fetchall()
    conn.close()
    return [{"sender_id": msg[0], "receiver_id": msg[1], "message": msg[2], "timestamp": msg[3]} for msg in messages]

@app.get("/")
def home():
    return {"message": "Travel Buddy ML API"}

def clean_destination(destination):
    if pd.isna(destination):
        return ""
    destination = str(destination).lower()
    destination = re.sub(r',\s*(india|usa|uk|united states|united kingdom|canada|australia|japan|china)$', '', destination)
    return destination.strip()

@app.post("/match")
def match_profiles(profile: Profile):
    try:
        # Load and prepare dataset
        df = pd.read_csv("profiles.csv")
        indian_profiles = generate_indian_profiles()
        international_profiles = generate_international_profiles()  # Add international profiles
        df = pd.concat([df, indian_profiles, international_profiles], ignore_index=True)
        df = df.fillna("")
        df["cleaned_destination"] = df["Destination"].apply(clean_destination)

        print(f"🔍 Filter: {profile.filter_type}, Destination: {profile.destination}, Date: {profile.travel_date}")

        # Apply filters
        if profile.filter_type == "destination":
            # SAME DESTINATION ONLY
            user_dest_clean = clean_destination(profile.destination)
            filtered_df = df[df["cleaned_destination"] == user_dest_clean]
            if len(filtered_df) == 0:
                filtered_df = df[df["cleaned_destination"].str.contains(user_dest_clean, na=False)]
            print(f"📍 Destination matches: {len(filtered_df)}")

        elif profile.filter_type == "dates":
            # SAME DATE + SAME DESTINATION
            if profile.travel_date:
                user_dest_clean = clean_destination(profile.destination)
                
                # First filter by destination
                destination_matches = df[df["cleaned_destination"] == user_dest_clean]
                if len(destination_matches) == 0:
                    destination_matches = df[df["cleaned_destination"].str.contains(user_dest_clean, na=False)]
                
                # Then filter by date
                def date_matches(row):
                    try:
                        if pd.isna(row.get('Start date')):
                            return False
                        start_date = pd.to_datetime(row['Start date']).date()
                        user_date = pd.to_datetime(profile.travel_date).date()
                        return start_date == user_date
                    except:
                        return False
                
                filtered_df = destination_matches[destination_matches.apply(date_matches, axis=1)]
                print(f"📅 Date+Destination matches: {len(filtered_df)}")
            else:
                filtered_df = df

        else:  # "all" - ALL USERS
            filtered_df = df
            print(f"🌟 All matches: {len(filtered_df)}")

        # If no matches
        if len(filtered_df) == 0:
            return {"matches": []}

        # Calculate compatibility
        filtered_df["combined"] = (
            filtered_df["Destination"].astype(str) + " " +
            filtered_df["Traveler nationality"].astype(str) + " " +
            filtered_df["Accommodation type"].astype(str) + " " +
            filtered_df["Transportation type"].astype(str) + " " +
            filtered_df.get("Travel style", "").astype(str)
        )

        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        X = vectorizer.fit_transform(filtered_df["combined"])
        user_text = f"{profile.destination} {profile.travel_style} {profile.hobbies}"
        user_vec = vectorizer.transform([user_text])
        scores = cosine_similarity(user_vec, X)[0]
        
        filtered_df = filtered_df.copy()
        filtered_df["compatibility"] = (scores * 100).round().astype(int)

        # Boost scores for exact destination matches
        user_dest_clean = clean_destination(profile.destination)
        exact_dest = filtered_df["cleaned_destination"] == user_dest_clean
        filtered_df.loc[exact_dest, "compatibility"] = filtered_df.loc[exact_dest, "compatibility"] + 30

        filtered_df["compatibility"] = filtered_df["compatibility"].clip(upper=100)

        # Get top matches (different count based on filter)
        if profile.filter_type == "all":
            top_matches = filtered_df.sort_values("compatibility", ascending=False).head(20)  # More for All filter
        else:
            top_matches = filtered_df.sort_values("compatibility", ascending=False).head(10)  # Less for specific filters

        # Prepare result with ALL necessary fields
        result = []
        for _, match in top_matches.iterrows():
            result.append({
                "Traveler name": match.get("Traveler name", "Unknown"),
                "Destination": match.get("Destination", ""),
                "Traveler nationality": match.get("Traveler nationality", ""),
                "Accommodation type": match.get("Accommodation type", ""),
                "Transportation type": match.get("Transportation type", ""),
                "Travel style": match.get("Travel style", ""),
                "Traveler age": match.get("Traveler age", 25),
                "Start date": match.get("Start date", ""),
                "End date": match.get("End date", ""),
                "Interests": match.get("Interests", ""),
                "compatibility": match.get("compatibility", 50)
            })

        print(f"✅ Returning {len(result)} matches")
        return {"matches": result}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"matches": []}

def generate_indian_profiles():
    """Generate 50 Indian travel profiles"""
    indian_destinations = ["Dehradun","Bageshwar","Goa", "Kerala", "Rajasthan", "Himachal Pradesh","Shimla","Nanital","Mussoorie","Lucknow","Delhi", "Mumbai", "Bangalore"]
    indian_names = ["Aarav Tyagi", "Priya Patel", "Rohan Singh", "Ananya Gupta", "Neha Kumar", "Arjun Mehta"]
    travel_styles = ["Adventurer", "Cultural", "Relaxed", "Foodie", "Spiritual", "Backpacker", "Beach Lover"]
    
    profiles = []
    for i in range(50):
        # Random dates within next 60 days
        start_days = random.randint(1, 60)
        end_days = start_days + random.randint(5, 14)
        start_date = (datetime.now() + timedelta(days=start_days)).strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=end_days)).strftime("%Y-%m-%d")
        
        profile = {
            "Trip ID": 1000 + i,
            "Destination": f"{random.choice(indian_destinations)}, India",
            "Start date": start_date,
            "End date": end_date,
            "Traveler name": random.choice(indian_names),
            "Traveler age": random.randint(22, 35),
            "Traveler nationality": "Indian",
            "Accommodation type": random.choice(["Hotel", "Hostel", "Resort", "Airbnb"]),
            "Transportation type": random.choice(["Flight", "Train", "Bus"]),
            "Travel style": random.choice(travel_styles),
            "Interests": random.choice([["Beaches", "Photography"], ["Temples", "Culture"], ["Food", "Shopping"]])
        }
        profiles.append(profile)
    
    return pd.DataFrame(profiles)

def generate_international_profiles():
    """Generate 50 International travel profiles"""
    international_destinations = ["Paris, France", "Tokyo, Japan", "Bali, Indonesia", "London, UK", "New York, USA", 
                                 "Sydney, Australia", "Bangkok, Thailand", "Dubai, UAE", "Singapore", "Rome, Italy"]
    international_names = ["Michael Brown", "Sophie Turner", "David Lee", "Emma Wilson", "James Smith", "Maria Garcia"]
    nationalities = ["American", "British", "Canadian", "Australian", "German", "French", "Japanese", "Korean"]
    
    profiles = []
    for i in range(50):
        start_days = random.randint(1, 60)
        end_days = start_days + random.randint(5, 14)
        start_date = (datetime.now() + timedelta(days=start_days)).strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=end_days)).strftime("%Y-%m-%d")
        
        profile = {
            "Trip ID": 2000 + i,
            "Destination": random.choice(international_destinations),
            "Start date": start_date,
            "End date": end_date,
            "Traveler name": random.choice(international_names),
            "Traveler age": random.randint(25, 45),
            "Traveler nationality": random.choice(nationalities),
            "Accommodation type": random.choice(["Hotel", "Resort", "Airbnb", "Guesthouse"]),
            "Transportation type": random.choice(["Flight", "Train", "Car"]),
            "Travel style": random.choice(["Adventurer", "Cultural", "Luxury", "Explorer"]),
            "Interests": random.choice([["Museums", "Art"], ["Hiking", "Nature"], ["Food", "Wine"], ["Shopping", "Nightlife"]])
        }
        profiles.append(profile)
    
    return pd.DataFrame(profiles)

# WebSocket and other endpoints...
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            save_message(user_id, message_data["receiver_id"], message_data["message"])
            await manager.send_personal_message(
                json.dumps({
                    "type": "message",
                    "sender_id": user_id,
                    "message": message_data["message"],
                    "timestamp": datetime.now().isoformat()
                }),
                message_data["receiver_id"]
            )
    except WebSocketDisconnect:
        manager.disconnect(user_id)

@app.get("/chat/history/{user1_id}/{user2_id}")
def get_chat_history_endpoint(user1_id: str, user2_id: str):
    return {"chat_history": get_chat_history(user1_id, user2_id)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)