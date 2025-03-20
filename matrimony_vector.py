# preprocess.py
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
from pymongo import MongoClient
# Load data
# df = pd.read_csv("data/users_data.csv")
# new_profile =    {
#     "Profile_ID": "N0004",
#     "Name": "Sohail",
#     "Gender": "Male",
#     "Age": 25,
#     "Marital_Status": "Single",
#     "Education": "Law",
#     "Profession": "Teacher",
#     "Sect": "Shia",
#     "Religious_Level": "Moderate",
#     "Country": "India",
#     "City": "Mumbai",
#     "Height_cm": 157,
#     "Skin_Tone": "Wheatish",
#     "Language": "Urdu",
#     "Financial_Status": "Middle Class",
#     "Expectations": "Caring",
#     "Hobbies": "Painting"
#   }
def create_vector(mongodb_uri,db_name="matrimonial",collection_name="whatsapp_data"):
    connect = MongoClient(mongodb_uri)
    # database name
    db = connect[db_name]

    # collection name or table
    collection = db[collection_name]
    # add new profile
    # collection.insert_one(new_profile)
    # find_user = collection.find_one({"Name": "Sohail"})
    # print(f"Recently Add {find_user}")
    # change the keys
    # collection.update_many(
    # {},  # Update all documents
    # { "$rename": { "Marital Status": "Marital_Status",
    #               "Height (cm)": "Height_cm",
    #               "Religious Level": "Religious_Level",
    #               "Financial Status":"Financial_Status"
    # } })
    # fetch data from Mongodb
    data = list(collection.find({}, {"_id":0}))
    

    # convert to Dataframe
    df = pd.DataFrame(data)
    # Update documents with missing fields
    # collection.update_many({"gender": {"$exists": False}}, {"$set": {"gender": "unknown"}})
    # collection.update_many({"native_place": {"$exists": False}}, {"$set": {"native_Place": "unknown"}})
    # collection.update_many({"preferences": {"$exists": False}}, {"$set": {"preferences": "unknown"}})
    # Fill missing values with an empty string or placeholder
    # df.fillna("unknown", inplace=True)

    required_fields = ["profile_id", "full_name", "date_of_birth", "age", "marital_status", 
                "religion", "education", "height", "native_place", "preferences"]

    # Fill missing fields with "unknown" before creating 'text' column
    for field in required_fields:
        if field not in df.columns:
            df[field] = "unknown"

    # df["text"] = (
    #     df["profile_id"].astype(str) + " " +
    #     df["full_name"].astype(str) + " " +
    #     "(Date Of Birth): "+ " " + df["date_of_birth"].astype(str) + " " +
    #     "(Age): "+ " " + df["age"].astype(str) + " " +
    #     "(Marital Status): "+ " " + df["marital_status"].astype(str) + " " +
    #     "(Religion): "+ " " + df["religion"].astype(str) + " " +
    #     "(Education): "+ " " + df["education"].astype(str) + " " +
    #     "(Height): "+ " " + df["height"].astype(str) + " " +
    #     "(Native_place): "+ " " + df["native_place"].astype(str) + " " +
    #     "(Preference): "+ " " + df["preferences"].astype(str)
    # )
    df["bio"] = (
        df["profile_id"].astype(str) + " \n" +
        df["full_name"].astype(str) + " \n" +
        "Date Of Birth: "+ " " + df["date_of_birth"].astype(str) + " \n" +
        "Age: "+ " " + df["age"].astype(str) + " \n" +
        "Marital Status: "+ " " + df["marital_status"].astype(str) + " \n" +
        "Gender: "+ " " + df["gender"].astype(str) + " \n" +
        "complexion: "+ " " + df["complexion"].astype(str) + " \n" +
        "Religion: "+ " " + df["religion"].astype(str) + " \n" +
        "Education: "+ " " + df["education"].astype(str) + " \n" +
        "Height: "+ " " + df["height"].astype(str) + " \n" +
        "Native_place: "+ " " + df["native_place"].astype(str) + " \n" +
        "residence: "+ " " + df["residence"].astype(str) + " \n" +
        "Father: "+ " " + df["father"].astype(str) + " \n" +
        "Mother: "+ " " + df["mother"].astype(str) + " \n" +
        "Maslak_sect: "+ " " + df["maslak_sect"].astype(str) + " \n" +
        "occupation: "+ " " + df["occupation"].astype(str) + " \n" +
        "Preference: "+ " " + df["preferences"].astype(str)
    )
    df["text"] = (
          df["profile_id"].astype(str) + " \n" +
        df["full_name"].astype(str) + " \n" +
        "Date Of Birth: "+ " " + df["date_of_birth"].astype(str) + " \n" +
        "Age: "+ " " + df["age"].astype(str) + " \n" +
        "Marital Status: "+ " " + df["marital_status"].astype(str) + " \n" +
        "Gender: "+ " " + df["gender"].astype(str) + " \n" +
        "complexion: "+ " " + df["complexion"].astype(str) + " \n" +
        "Religion: "+ " " + df["religion"].astype(str) + " \n" +
        "Education: "+ " " + df["education"].astype(str) + " \n" +
        "Height: "+ " " + df["height"].astype(str) + " \n" +
        "Native_place: "+ " " + df["native_place"].astype(str) + " \n" +
        "residence: "+ " " + df["residence"].astype(str) + " \n" +
        "Maslak_sect: "+ " " + df["maslak_sect"].astype(str) + " \n" +
        "occupation: "+ " " + df["occupation"].astype(str) + " \n" +
        "Preference: "+ " " + df["preferences"].astype(str)
    )
    # df["text"] = (
    #     "Preference: "+ " " + df["preferences"].astype(str)
    #     )

    # Convert the combined text to a list
    texts = df["text"].tolist()
    
    return df,data, texts

# Generate embeddings
def generate_embeddings(texts):
    if not texts:
        print("❌ Error: No text data available for embeddings.")
        return
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True)
    if len(embeddings) == 0:
        print("❌ Error: Embeddings could not be generated.")
        return
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype("float32")) # type: ignore
    # Save index
    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, "vectorstore/index.faiss")
    print("Vector store created and saved!")