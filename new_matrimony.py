import pickle
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
import os
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from new_matrimony_v import create_chunks,create_faiss_index
from langchain_core.prompts import ChatPromptTemplate
import pdb
# Load API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Load Gemini Model
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load MongoDB data and vector store
mongodb_uri = "mongodb://localhost:27017"  # Change this to your MongoDB URI
texts,df = create_chunks(mongodb_uri)

# this function generate emebeddings and insert into vectors
# create_faiss_index()

# print(df.head())  # Check first few rows
# print("Columns in df:", df.columns)
print(texts[0])
# generate_embeddings_with_clusters(df)
# Load API Key and Gemini Model
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to normalize embeddings
def normalize_embeddings(embeddings):
    return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

def search_profiles_with_gemini(df, user_name, top_k=5, location=None, education=None):
    """Find profiles using gender-based FAISS clustering and explain matches with Gemini."""

    # Get user profile
    user_profile = df[df["full_name"] == user_name]
    if user_profile.empty:
        return pd.DataFrame(), "❌ User not found."

    user_gender = user_profile.iloc[0]["gender"]

    # Select the appropriate FAISS index
    if user_gender == "Male":
        matched_df = df[df["gender"] == "Female"]  # Male searches for females
        index_path = "store/female_index.faiss"  # Male users search in the female index
        opposite_gender = "Female"
    elif user_gender== "Female":
        matched_df = df[df["gender"] == "Male"]  # Female searches for males
        index_path = "store/male_index.faiss"  # Female users search in the male index
        opposite_gender = "Male"
    else:
        return pd.DataFrame(), "❌ Invalid gender."

    print(f"🔍 User Gender: {user_gender}, Searching in: {index_path} (Looking for {opposite_gender})")

    # Load the FAISS index
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        return pd.DataFrame(), f"❌ Error loading FAISS index: {str(e)}"

    # Encode user profile text
    query_text = user_profile.iloc[0]["text"]
    query_embedding = model.encode([query_text]).astype("float32")
    query_embedding = normalize_embeddings(query_embedding)

    # Search FAISS
    distance, faiss_indices = index.search(query_embedding, k=top_k)  # Retrieve extra for filtering
    print(f"FAISS Retrieved Indices: {faiss_indices} and distance {distance} length of {len(faiss_indices[0])}")

    # Retrieve matched profiles
    # matched_profiles = df.iloc[faiss_indices[0]]
    pdb.set_trace()
    matched_profiles = matched_df.iloc[faiss_indices[0]]  # Ensure only opposite gender profiles are retrieved
    print(matched_profiles)
    # Apply additional filters
    if location:
        matched_profiles = matched_profiles[matched_profiles["native_place"].str.lower() == location.lower()]
    if education:
        matched_profiles = matched_profiles[matched_profiles["education"].str.lower() == education.lower()]

    matched_profiles = matched_profiles.head(top_k)  # Return top-k results

    if matched_profiles.empty:
        return pd.DataFrame(), "❌ No matches found."

    # Generate Explanation Using Gemini
    # combined_input = (
    #     "You are an AI-powered matchmaking assistant specializing in Muslim marriages. Your role is to find the most compatible matches based on religious alignment, user-defined preferences, and structured scoring. Strict cross-verification is required between the sought profile and the queried profile. The 'Preferences' section has the highest priority and must significantly impact the final score.\n\n"

    #     "Matchmaking priorities and weights:\n"
    #     "1. User Preferences with Cross Verification - 50%: Age, marital history, occupation, education, family background, career, culture, language, religious commitment, and lifestyle. Unmet mandatory preferences lower the score.\n"
    #     "2. Religious Alignment - 30%: Ensure sect alignment. Mismatches score below 50%.\n"
    #     "3. Personality & Lifestyle - 10%: Shared interests refine compatibility.\n"
    #     "4. Age Difference - 10%: Female age should be equal to or less than male’s unless flexibility is indicated.\n\n"

    #     "Final Match Score (%) = (User Preferences * 50%) + (Religious Alignment * 30%) + (Personality * 10%) + (Age * 10%). AI must provide a percentage with a breakdown. High scores require justification. Scores below 50% indicate incompatibility.\n\n"

    #     "Maintain fairness, transparency, and privacy. Do not force matches where religious alignment or key preferences do not match. The goal is to provide structured, ethical matchmaking aligned with Islamic principles.\n\n"

    #     f"User Profile: {query_text}\n\n"
    #     "These are the potential matches:\n"
    #     + "\n\n".join(matched_profiles["text"].tolist()) + "\n\n"
    #     "Your objective is to provide accurate, ethical, and structured matchmaking that aligns with Islamic principles while ensuring fairness and transparency in the scoring process."
    # )
    
    combined_input = (
        "You are an AI-powered matchmaking assistant specializing in Muslim marriages. Your role is to find the most compatible matches based on religious alignment, user-defined preferences, and structured scoring. Strict cross-verification is required between the sought profile and the queried profile. The 'Preferences' section has the highest priority and must significantly impact the final score."

            "Matchmaking priorities and weights:"

            "Show respectives scores against each listed item below without fail"
            "Most importantly look at the contents and context under preferences section and try matching *strictly* with that of opposite profile and vice a versa."
            "User Preferences with Cross Verification - 50%: *age*, *marital history*, *occupation* , *education*,*family background* *Native Place* *Maslak Sect*, career, culture, language, religious commitment, and lifestyle. Unmet mandatory preferences lower the score."
            "Religious Alignment - 30%: Ensure sect alignment. Mismatches score below 50%."
            "Personality & Lifestyle - 10%: Shared interests refine compatibility."
            "Age Difference - 10%: Female age should be equal to or less than male’s unless flexibility is indicated. calculate age with given date of birth if age is not available"
            "Final Match Score (%) = (User Preferences * 50%) + (Religious Alignment * 30%) + (Personality * 10%) + (Age * 10%). AI must provide a percentage with a breakdown. High scores require justification. Scores below 50% indicate incompatibility."

            "Maintain fairness, transparency, and privacy. Do not force matches where religious alignment or key preferences do not match. The goal is to provide structured, ethical matchmaking aligned with Islamic principles."

            f"User Profile: {query_text}\n\n"
            "These are the potential matches:\n"
            + "\n\n".join(matched_profiles["bio"].tolist()) + "\n\n"
            "Objective Your goal is to provide accurate, ethical, and structured matchmaking that aligns with Islamic principles while ensuring fairness and transparency in the scoring process. The Preferences section is given the highest weight to reflect user expectations accurately."
        )

    print(combined_input)
    # Send to Gemini LLM
    messages = [
        SystemMessage(content="You are an AI assistant that helps match profiles for a matrimonial platform."),
        HumanMessage(content=combined_input)
    ]

    result = gemini_model.invoke(messages)

    # Display Matches with Explanation
    print("\n--- Gemini's Response on Matches ---")
    print(result.content)
    
    return matched_profiles, result.content

def transform_llm_response(llm_response):
    gemini_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    # messages = [
    # ("system", "You are an AI assisstence"),
    # ("human", """From the following text, prepare a dictionary that contains following keys:
    #  Profile, Age, Marital History, Occupation, Preferences, Matched_Profile. Prepare nested dictionaries under Age, Marital History,
    #  Occupation and Matched_Profile keys and have two more keys namely score and remarks.
    #  In remarks keep everything apart from score and age. Generate a list as the value of Matched_Profile key which should further contain a dictionary of matched profile that includes all the above keys.
    #  If the text does not contain specific score, consider it as 'N/A'. All the mentioned keys must be available in the final dictinoary.
    #  The text is as follows: {text}""")]
    
    # prompt_template = ChatPromptTemplate.from_messages(messages)
    # prompt = prompt_template.invoke({"text": llm_response})
    # result = gemini_model.invoke(prompt)
    # Define the user profile schema
    class UserProfile(BaseModel):
        name: str
        age_range: str
        marital_status: str
        religion: str
        location: str
        education: str
        preferences: str

    # Define a model for Match Evaluation scores
    class MatchScore(BaseModel):
        user_preferences: int
        religious_alignment: int
        personality_lifestyle: int
        age: int
        total_score: int
        compatibility: str

    # Define a model for each match
    class Match(BaseModel):
        profile_id: int = Field(description="Exctract profile_id")
        name: str
        age: int
        marital_status: str
        occupation: str
        education: str
        family_background: Optional[str] = "Unknown"
        native_place: str
        maslak_sect: Optional[str] = Field(description="Write only the maslak or sect if available", default="Unknown")
        religious_alignment: Optional[str] = "Unknown"
        personality_lifestyle: Optional[str] = "Unknown"
        preferences: str
        score_breakdown: MatchScore

    # Define a model for the overall match analysis
    class MatchAnalysis(BaseModel):
        user_profile: UserProfile
        matches: List[Match]
        conclusion: str
    
    structured_model = gemini_model.with_structured_output(MatchAnalysis)
    
    result = structured_model.invoke(llm_response)
    result_dict = dict(result)
    
    return result_dict
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

    matches_users,llm_matches= search_profiles_with_gemini(df, user_name, top_k=top_k, location=chosen_location, education=chosen_education)
    final = transform_llm_response(llm_matches)
    print(final)
    print(f"type of final output : {type(final)}")
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
        # ranked_matches = extract_top_llm_matches_by_id(llm_matches, matches_users)

        # print("\n⭐ Most Matched Profiles Based on LLM Feedback ⭐")
        # for rank, (idx, profile_id, score) in enumerate(ranked_matches, start=1):
        #     profile = matches_users.loc[idx]
        #     print(f"\nRank #{rank} - ID: {profile_id} (LLM Score: {score})")
        #     for key in ["full_name","gender", "age", "native_place", "education", "preferences"]:
        #         print(f"{key.capitalize()}: {profile[key]}")
        #     print("-" * 30)
    i += 1

