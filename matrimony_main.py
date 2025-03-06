from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
import os
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from matrimony_vector import create_vector, generate_embeddings
import pdb
# Load API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load Gemini Model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load MongoDB data and vector store
mongodb_uri = "mongodb://localhost:27017"  # Change this to your MongoDB URI
df, data, texts = create_vector(mongodb_uri)
# print(df.head())  # Check first few rows
# print("Columns in df:", df.columns)
print(texts[0])

# example of chunking: Preference:  well educated, well settled Job, decent family, graduation compulsory. 5 times salah  only fatiha followers,  
# no bad habits(KSA) Please excuse us.
generate_embeddings(texts)

# Function to normalize embeddings
def normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def search_profiles_with_gemini(df, user_name, top_k=5, location=None, education=None, preferences=None):
    """Find similar profiles using FAISS and explain matches with Gemini."""
    
    # Load FAISS index
    index = faiss.read_index("vectorstore/index.faiss")
    try:
        user_index = df[df["full_name"] == user_name].index[0]
    except IndexError:
        print("❌ User not found in the database.")
        return pd.DataFrame(), []

    print(df.iloc[user_index]["text"])
    print(df.iloc[user_index]["bio"])
    query_text = df.iloc[user_index]["text"]
    query_embedding = model.encode([query_text]).astype("float32")
    query_embedding = normalize_embeddings(query_embedding)

    # Perform FAISS Similarity Search (Now using Inner Product instead of L2)
    scores, indices = index.search(query_embedding, k=top_k * 3)  # Fetch more results
    print(indices)
    # Convert to cosine similarity (since normalized, IP is equivalent to cosine)
    results_with_scores = list(zip(indices[0], scores[0]))

    # Sort in descending order (higher similarity = better match)
    results_with_scores.sort(key=lambda x: x[1], reverse=True)

    # Filter gender-based matches
    selected_gender = df.iloc[user_index]["gender"]
    opposite_gender = "Female" if selected_gender == "Male" else "Male"
    # pdb.set_trace()
    filtered_results = []
    # print(f"Checking user {user_name} & gender {opposite_gender}")
    for i, score in results_with_scores:
        # print(f"iteration {i}: full_name {df.iloc[i]['full_name']} and gender {df.iloc[i]['gender']}")
        if df.iloc[i]["full_name"] != user_name and df.iloc[i]["gender"] == opposite_gender:
            filtered_results.append((i, score))
    
    # ]
    # filtered_results = [
    #     (i, score) for i, score in results_with_scores
    #     if df.iloc[i]["full_name"] != user_name and df.iloc[i]["gender"] == opposite_gender
    # ]
    # print("Filtered Results:", filtered_results)
    # pdb.set_trace()
    try:
        if len(filtered_results) != 0:
            # Create DataFrame for matched profiles with similarity scores
            matched_df = pd.DataFrame([
                {**df.iloc[i].to_dict(), "score": score} for i, score in filtered_results
            ])
            # pdb.set_trace()
            # Apply additional filters for location and education if provided
            if location:
                matched_df = matched_df[matched_df["native_place"].str.lower() == location.lower()]
            if education:
                matched_df = matched_df[matched_df["education"].str.lower() == education.lower()]
            # pdb.set_trace()
            # Sort by similarity score (higher is better)
            # matched_df = matched_df.sort_values(by="score", ascending=False)
            if "score" in matched_df.columns:
                matched_df = matched_df.sort_values(by="score", ascending=False)
            else:
                print("⚠️ Warning: 'score' column missing. Skipping sorting step.")

            # Select Top K Matches
            final_matches = matched_df.head(top_k)
            final_matches["text"].tolist()
            # Generate Explanation Using Gemini
            # combined_input = (
            #     "Here is the user profile, and they are looking for their future partner. "
            #     "Can you help them find a match based on preferences like place, education, complexion, Height etc.?\n\n"
            #     f"User Profile: {query_text}\n\n"
            #     "These are the potential matches:\n"
            #     + "\n\n".join(final_matches["text"].tolist())
            #     + "\n\nFor each profile, explain why it is a good match or why it is not a good match."
            # )
            # combined_input = (
            #     "Here is the user profile, and they are looking for their future partner. "
            #     "Can you help them find a match based on preferences like place, education, complexion, Height etc.?\n\n"
            #     " if user have no Demand and according to his/her age, education, height, complexion, Religion etc find a matches\n\n"
            #     f"User Profile: {df.iloc[user_index]['bio']}\n\n"
            #     "These are the potential matches:\n"
            #     + "\n\n".join(final_matches["bio"].tolist())+ "\n\n"
            #     + "\n\nFor each profile, explain why it is a good match or why it is not a good match."
            # )
            combined_input = (
                "You are a great match maker, every boy and every girl wants their perfect match based on the eachother's profile."
                "Check on every detail of query profile and match it with that of the opposite profiles."
                "Most importantly look at the contents and context under preferences section and try matching *strictly* with that of opposite profile and vice a versa."
                "And finally provide us most matching profiles along with 1. Summary why is the match. \n"
                "2. What potential factors influenced the match 3. An output line stating what degree of match is accomplished among both the profiles viz., Excellent match, Good Match, Possible Match and Poor Match. \n\n"
                f"User Profile: {query_text}\n\n"
                "These are the potential matches:\n"
                + "\n\n".join(final_matches["bio"].tolist())+ "\n\n"
                + "\n\nFor each profile, explain why it is a good match or why it is not a good match."
            )
            # combined_input = (
            #     "You are a great match maker, every boy and every girl wants their perfect match based on the eachother's profile. Check on every detail of query profile and match it with that of the opposite profiles. Most importantly look at the contents and context under preferences section and try matching with that of opposite profile and vice a versa. and finally provide us most matching profiles along with 1. Summary why is the match 2. What potential factors influenced the match 3. An output line stating what degree of match is accomplished among both the profiles viz., Excellent match, Good Match, Possible Match and Poor Match. \n\n"
            #     f"User Profile: {query_text}\n\n"
            #     "These are the potential matches:\n"
            #     + "\n\n".join(final_matches["bio"].tolist())+ "\n\n"
            #     + "\n\nFor each profile, explain why it is a good match or why it is not a good match."
            # )
            # print(combined_input)
            # Send to Gemini LLM
            messages = [
                SystemMessage(content="You are an AI assistant that helps match profiles for a matrimonial platform."),
                HumanMessage(content=combined_input)
            ]

            result = gemini_model.invoke(messages)

            # Display Matches with Explanation
            print("\n--- Gemini's Response on Matches ---")
            print(result.content)
        
            return final_matches, result.content
        else:
            final_matches = pd.DataFrame()
            return final_matches, "empty"
    except Exception as e:
        print(str(e))
        # pdb.set_trace()

import re

# Extract top matches based on IDs mentioned in LLM response
def extract_top_llm_matches_by_id(llm_matches, matches_users):
    """Analyze LLM response and extract top matches using profile IDs."""
    scores = []

    for idx, profile in matches_users.iterrows():
        profile_id = profile["full_name"]  # Use ID instead of name
        match = re.search(rf"ID:\s*{profile_id}.*?(good match|best match|high compatibility)", llm_matches, re.IGNORECASE)
        score = 2 if match else 1  # Assign higher score if explicitly mentioned
        
        scores.append((idx, profile_id, score))

    # Sort by LLM ranking score
    scores.sort(key=lambda x: x[2], reverse=True)

    return scores

i = 1

while i < 10:
    # Example Search with Gemini Justification
    user_name = input("Enter your name: ").strip()
        # Find user profile safely
    user_profile = df[df["full_name"] == user_name]

    if user_profile.empty:
        print("❌ User not found. Exiting...")
        exit()

    # Get user details safely
    user_profile = user_profile.iloc[0]  # Convert to Series
    # selected_keys = ["name", "age", "native_place", "education", "preferences"]
    # print("\nYour Profile:\n" + "=" * 20)
    # for key, value in user_profile.items():
    #     print(f"{key:20}: {value}")
    selected_keys = ["profile_id","full_name", "gender" "age", "native_place", "education", "preferences"]

    print("\nYour Profile:\n" + "=" * 20)
    for key in selected_keys:
        if key in user_profile:
            print(f"{key:20}: {user_profile[key]}")
    top_k = int(input("How many matches? "))
    chosen_location = input("Filter by location (or press Enter to skip): ").strip()
    chosen_education = input("Filter by education (or press Enter to skip): ").strip()

    matches_users,llm_matches= search_profiles_with_gemini(df, user_name, top_k=top_k+1, location=chosen_location, education=chosen_education)

    if matches_users.empty:
        print("❌ No matching profiles found.")
    else:
        print("\nTop Matched Profiles:\n" + "*" * 30)
        # Get user details safely
        matches_profile = matches_users.iloc[0]  # Convert to Series
        selected_keys = ["profile_id","full_name", "gender", "age", "native_place", "education", "preferences"]
        for idx, profile in matches_users.iterrows():
            # print("\nProfile #" + str(idx + 1))
            print("-" * 30)
            for key in selected_keys:
                if key in profile:
                    print(f"{key:20}: {profile[key]}")
        # Get top LLM-ranked matches by ID
        ranked_matches = extract_top_llm_matches_by_id(llm_matches, matches_users)

        print("\n⭐ Most Matched Profiles Based on LLM Feedback ⭐")
        for rank, (idx, profile_id, score) in enumerate(ranked_matches, start=1):
            profile = matches_users.loc[idx]
            print(f"\nRank #{rank} - ID: {profile_id} (LLM Score: {score})")
            for key in ["full_name","gender", "age", "native_place", "education", "preferences"]:
                print(f"{key.capitalize()}: {profile[key]}")
            print("-" * 30)
    i += 1



