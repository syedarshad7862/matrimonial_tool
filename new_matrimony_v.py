import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
import os

def create_chunks(mongodb_uri, db_name="matrimonial", collection_name="new_profiles"):
    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Fetch data from MongoDB
    data = list(collection.find({}, {"_id": 0}))  # Exclude MongoDB ID
    df = pd.DataFrame(data)

    # Ensure required fields exist
    required_fields = ["pref_age_range", "pref_marital_status", "pref_complexion", "pref_education", "pref_height", 
                    "pref_native_place", "pref_maslak_sect", "pref_no_of_siblings", "pref_work_job", "pref_go_to_dargah", "pref_mother_tongue", "pref_deendari","profile_id","sect" "religious_practice", "full_name", "date_of_birth", "age", "marital_status", 
                "religion", "education", "height", "native_place",'occupation','preferences']
    for field in required_fields:
        if field not in df.columns:
            df[field] = "unknown"
    # pk
    df["text"] = (
        df["full_name"].astype(str) + " \n" +
        "age_range: "+ " " + df["pref_age_range"].astype(str) + " \n" +
        "Marital Status: "+ " " + df["pref_marital_status"].astype(str) + " \n" +
        "Complexion: "+ " " + df["pref_complexion"].astype(str) + " \n" +
        "Education: "+ " " + df["pref_education"].astype(str) + " \n" +
        "Height: "+ " " + df["pref_height"].astype(str) + " \n" +
        "Native_place: "+ " " + df["pref_native_place"].astype(str) + " \n" +
        "Maslak_sect: "+ " " + df["pref_maslak_sect"].astype(str) + " \n" +
        "Siblings: "+ " " + df["pref_no_of_siblings"].astype(str) + " \n" +
        "Occupation: "+ " " + df["pref_work_job"].astype(str) + " \n" +
        "Go to dargah: "+ " " + df["pref_go_to_dargah"].astype(str) + " \n" +
        "Mother tongue: "+ " " + df["pref_mother_tongue"].astype(str) + " \n" +
        "Deender: "+ " " + df["pref_deendari"].astype(str) + " \n" +
        "location: "+ " " + df["pref_location"].astype(str) + " \n" +
        "sect: "+ " " + df["sect"].astype(str) + " \n" +
        "religious_practice: "+ " " + df["religious_practice"].astype(str) + " \n" +
        "pref_own_house: "+ " " + df["pref_own_house"].astype(str) + " \n" +
        "Preferences: "+ " " + df["preferences"].astype(str)
    )
    # normal keys
    # df["text"] = (
    #     df["profile_id"].astype(str) + " \n" +
    #     df["full_name"].astype(str) + " \n" +
    #     "Date Of Birth: "+ " " + df["date_of_birth"].astype(str) + " \n" +
    #     "Age: "+ " " + df["age"].astype(str) + " \n" +
    #     "Marital Status: "+ " " + df["marital_status"].astype(str) + " \n" +
    #     "Gender: "+ " " + df["gender"].astype(str) + " \n" +
    #     "complexion: "+ " " + df["complexion"].astype(str) + " \n" +
    #     "Education: "+ " " + df["education"].astype(str) + " \n" +
    #     "Height: "+ " " + df["height"].astype(str) + " \n" +
    #     "Native_place: "+ " " + df["native_place"].astype(str) + " \n" +
    #     "residence: "+ " " + df["residence"].astype(str) + " \n" +
    #     "Father: "+ " " + df["father"].astype(str) + " \n" +
    #     "Mother: "+ " " + df["mother"].astype(str) + " \n" +
    #     "Maslak_sect: "+ " " + df["sect"].astype(str) + " \n" +
    #     "occupation: "+ " " + df["occupation"].astype(str) + " \n" +
    #     "Preference: "+ " " + df["preferences"].astype(str)
    # )
    # nk + pk 
    # df["text"] = (
    #     df["profile_id"].astype(str) + " \n" +
    #     df["full_name"].astype(str) + " \n" +
    #     "Date Of Birth: "+ " " + df["date_of_birth"].astype(str) + " \n" +
    #     "Age: "+ " " + df["age"].astype(str) + " \n" +
    #     "Marital Status: "+ " " + df["marital_status"].astype(str) + " \n" +
    #     "Gender: "+ " " + df["gender"].astype(str) + " \n" +
    #     "complexion: "+ " " + df["complexion"].astype(str) + " \n" +
    #     "Education: "+ " " + df["education"].astype(str) + " \n" +
    #     "Height: "+ " " + df["height"].astype(str) + " \n" +
    #     "Native_place: "+ " " + df["native_place"].astype(str) + " \n" +
    #     "residence: "+ " " + df["residence"].astype(str) + " \n" +
    #     "Father: "+ " " + df["father"].astype(str) + " \n" +
    #     "Mother: "+ " " + df["mother"].astype(str) + " \n" +
    #     "Maslak_sect: "+ " " + df["sect"].astype(str) + " \n" +
    #     "occupation: "+ " " + df["occupation"].astype(str) + " \n" +
    #     "age_range: "+ " " + df["pref_age_range"].astype(str) + " \n" +
    #     "Marital Status: "+ " " + df["pref_marital_status"].astype(str) + " \n" +
    #     "Complexion: "+ " " + df["pref_complexion"].astype(str) + " \n" +
    #     "Education: "+ " " + df["pref_education"].astype(str) + " \n" +
    #     "Height: "+ " " + df["pref_height"].astype(str) + " \n" +
    #     "Native_place: "+ " " + df["pref_native_place"].astype(str) + " \n" +
    #     "Maslak_sect: "+ " " + df["pref_maslak_sect"].astype(str) + " \n" +
    #     "Siblings: "+ " " + df["pref_no_of_siblings"].astype(str) + " \n" +
    #     "Occupation: "+ " " + df["pref_work_job"].astype(str) + " \n" +
    #     "Go to dargah: "+ " " + df["pref_go_to_dargah"].astype(str) + " \n" +
    #     "Mother tongue: "+ " " + df["pref_mother_tongue"].astype(str) + " \n" +
    #     "Deender: "+ " " + df["pref_deendari"].astype(str) + " \n" +
    #     "location: "+ " " + df["pref_location"].astype(str) + " \n" +
    #     "sect: "+ " " + df["sect"].astype(str) + " \n" +
    #     "religious_practice: "+ " " + df["religious_practice"].astype(str) + " \n" +
    #     "pref_own_house: "+ " " + df["pref_own_house"].astype(str) + " \n" +
    #     "Preferences: "+ " " + df["preferences"].astype(str)
    # )
    df["bio"] = (
        df["profile_id"].astype(str) + " \n" +
        df["full_name"].astype(str) + " \n" +
        "Date Of Birth: "+ " " + df["date_of_birth"].astype(str) + " \n" +
        "Age: "+ " " + df["age"].astype(str) + " \n" +
        "Marital Status: "+ " " + df["marital_status"].astype(str) + " \n" +
        "Gender: "+ " " + df["gender"].astype(str) + " \n" +
        "complexion: "+ " " + df["complexion"].astype(str) + " \n" +
        "Education: "+ " " + df["education"].astype(str) + " \n" +
        "Height: "+ " " + df["height"].astype(str) + " \n" +
        "Native_place: "+ " " + df["native_place"].astype(str) + " \n" +
        "residence: "+ " " + df["residence"].astype(str) + " \n" +
        "Father: "+ " " + df["father"].astype(str) + " \n" +
        "Mother: "+ " " + df["mother"].astype(str) + " \n" +
        "Maslak_sect: "+ " " + df["sect"].astype(str) + " \n" +
        "occupation: "+ " " + df["occupation"].astype(str) + " \n" +
        "Preference: "+ " " + df["preferences"].astype(str)
    )
    # Convert the combined text to a list
    texts = df["text"].tolist()
    return texts,df


# def create_faiss_index(df):
#     """Create separate FAISS indexes for male and female profiles."""
    
#     # Load Sentence Transformer Model
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     print("Before FAISS Encoding:")
#     # Normalize Gender Labels
#     df["gender"] = df["gender"].str.strip().str.lower()  # Convert to lowercase and remove spaces

#     # Map variations to standard values
#     gender_mapping = {
#         "female": "Female",
#         "male": "Male",
#         "female ": "Female",
#         "male ": "Male",
#         "female.": "Female",
#         "male.": "Male",
#         "femail": "Female",
#         "femaile": "Female",
#         "femlae": "Female",
#         "famele": "Female",
#         "fem": "Female",
#         "f": "Female",
#         "m": "Male",
#         "only for ladies": "Female",
#         "nil": "Unknown",  # Mark "Nil" as Unknown
#     }

#     # Apply mapping
#     df["gender"] = df["gender"].replace(gender_mapping)

#     # Keep only Male and Female
#     df = df[df["gender"].isin(["Male", "Female"])]

#     print("After Gender Normalization:")
#     # print(df["gender"].value_counts())

#     print(df["gender"].value_counts())
    

#     # Separate male and female profiles
#     male_profiles = df[df["gender"] == "Male"]
#     female_profiles = df[df["gender"] == "Female"]
#     print(f"Male Profiles Count: {len(male_profiles)}")
#     print(f"Female Profiles Count: {len(female_profiles)}")
    
#     # Encode profiles
#     male_embeddings = model.encode(male_profiles["text"].tolist()).astype("float32")
#     female_embeddings = model.encode(female_profiles["text"].tolist()).astype("float32")

#     # Normalize embeddings
#     male_embeddings /= np.linalg.norm(male_embeddings, axis=1, keepdims=True)
#     female_embeddings /= np.linalg.norm(female_embeddings, axis=1, keepdims=True)

#     # Create FAISS indexes
#     male_index = faiss.IndexFlatL2(male_embeddings.shape[1])
#     female_index = faiss.IndexFlatL2(female_embeddings.shape[1])

#     male_index.add(male_embeddings)
#     female_index.add(female_embeddings)

#     os.makedirs("newvectorstore", exist_ok=True)
#     # Save indexes
#     faiss.write_index(male_index, "newvectorstore/male_index.faiss")
#     faiss.write_index(female_index, "newvectorstore/female_index.faiss")

#     print("✅ FAISS indexes created successfully.")

# def create_faiss_index(df):
#     """Create separate FAISS indexes for male and female profiles."""

#     model = SentenceTransformer("all-MiniLM-L6-v2")

#     # Normalize gender labels before FAISS indexing
#     # df["gender"] = df["gender"].str.strip().str.lower()

#     # gender_mapping = {
#     #     "female": "Female",
#     #     "male": "Male",
#     #     "f": "Female",
#     #     "m": "Male",
#     #     "only for ladies": "Female",
#     #     "nil": "Unknown"
#     # }
    
#     # df["gender"] = df["gender"].replace(gender_mapping)
#     # df = df[df["gender"].isin(["Male", "Female"])]  # Keep only valid profiles

#     # Separate Male & Female profiles **BEFORE** embedding
#     male_profiles = df[df["gender"].str.lower() == "male"]
#     female_profiles = df[df["gender"].str.lower() == "female"]

#     print(f"✔ Male Profiles: {len(male_profiles)}")
#     print(f"✔ Female Profiles: {len(female_profiles)}")

#     # Convert text to embeddings
#     male_embeddings = model.encode(male_profiles["text"].tolist()).astype("float32")
#     female_embeddings = model.encode(female_profiles["text"].tolist()).astype("float32")

#     # Normalize embeddings
#     male_embeddings /= np.linalg.norm(male_embeddings, axis=1, keepdims=True)
#     female_embeddings /= np.linalg.norm(female_embeddings, axis=1, keepdims=True)

#     # Create FAISS indexes **SEPARATELY**
#     male_index = faiss.IndexFlatL2(male_embeddings.shape[1])
#     female_index = faiss.IndexFlatL2(female_embeddings.shape[1])

#     male_index.add(male_embeddings)  # Male Index (to match Females)
#     female_index.add(female_embeddings)  # Female Index (to match Males)

#     os.makedirs("newvectorstore", exist_ok=True)

#     # Save FAISS indexes separately
#     faiss.write_index(male_index, "newvectorstore/male_index.faiss")   # Male users will search here (matches Female)
#     faiss.write_index(female_index, "newvectorstore/female_index.faiss")  # Female users will search here (matches Male)

#     print("✅ FAISS indexes created successfully.")
# Example usage
# create_faiss_index("mongodb://localhost:27017", "matrimonial", "whatsapp_data")

def create_faiss_index(mongodb_uri = "mongodb://localhost:27017", db_name="matrimonial", collection_name="new_profiles"):
    """Create separate FAISS indexes for male and female profiles."""

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Connect to MongoDB
    client = MongoClient(mongodb_uri)
    db = client[db_name]
    collection = db[collection_name]

    # Fetch data from MongoDB
    male_data = list(collection.find({"gender": "Male"}, {"_id": 0}))  # Exclude MongoDB ID
    female_data = list(collection.find({"gender": "Female"}, {"_id": 0}))  # Exclude MongoDB ID
    male_df = pd.DataFrame(male_data)
    female_df = pd.DataFrame(female_data)
    
    # Ensure required fields exist
    required_fields = ["pref_age_range", "pref_marital_status", "pref_complexion", "pref_education", "pref_height", 
                    "pref_native_place", "pref_maslak_sect", "pref_no_of_siblings", "pref_work_job", "pref_go_to_dargah", "pref_mother_tongue", "pref_deendari","profile_id","sect", "full_name", "date_of_birth", "age", "marital_status", 
                "religion", "education","maslak_sect" "height","religious_practice" ,"native_place",'occupation','preferences',"go_to_dargah"]
    for field in required_fields:
        if field not in male_df.columns:
            male_df[field] = "unknown"
    for field in required_fields:
        if field not in female_df.columns:
            female_df[field] = "unknown"
    #   pref keys      
    # male_df["text"] = (
        # "age_range: "+ " " + male_df["pref_age_range"].astype(str) + " \n" +
        # "Marital Status: "+ " " + male_df["pref_marital_status"].astype(str) + " \n" +
        # "Complexion: "+ " " + male_df["pref_complexion"].astype(str) + " \n" +
        # "Education: "+ " " + male_df["pref_education"].astype(str) + " \n" +
        # "Height: "+ " " + male_df["pref_height"].astype(str) + " \n" +
        # "Native_place: "+ " " + male_df["pref_native_place"].astype(str) + " \n" +
        # "Maslak_sect: "+ " " + male_df["pref_maslak_sect"].astype(str) + " \n" +
        # "Siblings: "+ " " + male_df["pref_no_of_siblings"].astype(str) + " \n" +
        # "Occupation: "+ " " + male_df["pref_work_job"].astype(str) + " \n" +
        # "Go to dargah: "+ " " + male_df["pref_go_to_dargah"].astype(str) + " \n" +
        # "Mother tongue: "+ " " + male_df["pref_mother_tongue"].astype(str) + " \n" +
        # "Deender: "+ " " + male_df["pref_deendari"].astype(str) + " \n" +
        # "location: "+ " " + male_df["pref_location"].astype(str) + " \n" +
        # "pref_own_house: "+ " " + male_df["pref_own_house"].astype(str) + " \n" +
        # "Preferences: "+ " " + male_df["preferences"].astype(str)
    # )
    # female_df["text"] = (
        # "age_range: "+ " " + male_df["pref_age_range"].astype(str) + " \n" +
        # "Marital Status: "+ " " + male_df["pref_marital_status"].astype(str) + " \n" +
        # "Complexion: "+ " " + male_df["pref_complexion"].astype(str) + " \n" +
        # "Education: "+ " " + male_df["pref_education"].astype(str) + " \n" +
        # "Height: "+ " " + male_df["pref_height"].astype(str) + " \n" +
        # "Native_place: "+ " " + male_df["pref_native_place"].astype(str) + " \n" +
        # "Maslak_sect: "+ " " + male_df["pref_maslak_sect"].astype(str) + " \n" +
        # "Siblings: "+ " " + male_df["pref_no_of_siblings"].astype(str) + " \n" +
        # "Occupation: "+ " " + male_df["pref_work_job"].astype(str) + " \n" +
        # "Go to dargah: "+ " " + male_df["pref_go_to_dargah"].astype(str) + " \n" +
        # "Mother tongue: "+ " " + male_df["pref_mother_tongue"].astype(str) + " \n" +
        # "Deender: "+ " " + male_df["pref_deendari"].astype(str) + " \n" +
        # "location: "+ " " + male_df["pref_location"].astype(str) + " \n" +
        # "pref_own_house: "+ " " + male_df["pref_own_house"].astype(str) + " \n" +
        # "Preferences: "+ " " + male_df["preferences"].astype(str)
    # )
    # normal keys
    male_df["text"] = (
        male_df["profile_id"].astype(str) + " \n" +
        male_df["full_name"].astype(str) + " \n" +
        "Date Of Birth: "+ " " + male_df["date_of_birth"].astype(str) + " \n" +
        "Age: "+ " " + male_df["age"].astype(str) + " \n" +
        "Marital Status: "+ " " + male_df["marital_status"].astype(str) + " \n" +
        "Gender: "+ " " + male_df["gender"].astype(str) + " \n" +
        "complexion: "+ " " + male_df["complexion"].astype(str) + " \n" +
        "Education: "+ " " + male_df["education"].astype(str) + " \n" +
        "Height: "+ " " + male_df["height"].astype(str) + " \n" +
        "Native_place: "+ " " + male_df["native_place"].astype(str) + " \n" +
        "residence: "+ " " + male_df["residence"].astype(str) + " \n" +
        "Father: "+ " " + male_df["father"].astype(str) + " \n" +
        "Mother: "+ " " + male_df["mother"].astype(str) + " \n" +
        "sect: "+ " " + male_df["sect"].astype(str) + " \n" +
        "religious_practice: "+ " " + male_df["religious_practice"].astype(str) + " \n" +
        "go_to_dargah: "+ " " + male_df["go_to_dargah"].astype(str) + " \n" +
        "occupation: "+ " " + male_df["occupation"].astype(str) + " \n" +
        "Preference: "+ " " + male_df["preferences"].astype(str)
    )
    female_df["text"] = (
        female_df["profile_id"].astype(str) + " \n" +
        female_df["full_name"].astype(str) + " \n" +
        "Date Of Birth: "+ " " + female_df["date_of_birth"].astype(str) + " \n" +
        "Age: "+ " " + female_df["age"].astype(str) + " \n" +
        "Marital Status: "+ " " + female_df["marital_status"].astype(str) + " \n" +
        "Gender: "+ " " + female_df["gender"].astype(str) + " \n" +
        "complexion: "+ " " + female_df["complexion"].astype(str) + " \n" +
        "Education: "+ " " + female_df["education"].astype(str) + " \n" +
        "Height: "+ " " + female_df["height"].astype(str) + " \n" +
        "Native_place: "+ " " + female_df["native_place"].astype(str) + " \n" +
        "residence: "+ " " + female_df["residence"].astype(str) + " \n" +
        "Father: "+ " " + female_df["father"].astype(str) + " \n" +
        "Mother: "+ " " + female_df["mother"].astype(str) + " \n" +
        "sect: "+ " " + female_df["sect"].astype(str) + " \n" +
        "religious_practice,: "+ " " + female_df["religious_practice"].astype(str) + " \n" +
        "go_to_dargah: "+ " " + female_df["go_to_dargah"].astype(str) + " \n" +
        "occupation: "+ " " + female_df["occupation"].astype(str) + " \n" +
        "Preference: "+ " " + female_df["preferences"].astype(str)
    )
    # normal keys + pref keys
    # male_df["text"] = (
    #     male_df["profile_id"].astype(str) + " \n" +
    #     male_df["full_name"].astype(str) + " \n" +
    #     "Date Of Birth: "+ " " + male_df["date_of_birth"].astype(str) + " \n" +
    #     "Age: "+ " " + male_df["age"].astype(str) + " \n" +
    #     "Marital Status: "+ " " + male_df["marital_status"].astype(str) + " \n" +
    #     "Gender: "+ " " + male_df["gender"].astype(str) + " \n" +
    #     "complexion: "+ " " + male_df["complexion"].astype(str) + " \n" +
    #     "Education: "+ " " + male_df["education"].astype(str) + " \n" +
    #     "Height: "+ " " + male_df["height"].astype(str) + " \n" +
    #     "Native_place: "+ " " + male_df["native_place"].astype(str) + " \n" +
    #     "residence: "+ " " + male_df["residence"].astype(str) + " \n" +
    #     "Father: "+ " " + male_df["father"].astype(str) + " \n" +
    #     "Mother: "+ " " + male_df["mother"].astype(str) + " \n" +
    #     "sect: "+ " " + male_df["sect"].astype(str) + " \n" +
    #     "religious_practice: "+ " " + male_df["religious_practice"].astype(str) + " \n" +
    #     "go_to_dargah: "+ " " + male_df["go_to_dargah"].astype(str) + " \n" +
    #     "occupation: "+ " " + male_df["occupation"].astype(str) + " \n" +
    #     "age_range: "+ " " + male_df["pref_age_range"].astype(str) + " \n" +
    #     "Marital Status: "+ " " + male_df["pref_marital_status"].astype(str) + " \n" +
    #     "Complexion: "+ " " + male_df["pref_complexion"].astype(str) + " \n" +
    #     "Education: "+ " " + male_df["pref_education"].astype(str) + " \n" +
    #     "Height: "+ " " + male_df["pref_height"].astype(str) + " \n" +
    #     "Native_place: "+ " " + male_df["pref_native_place"].astype(str) + " \n" +
    #     "Maslak_sect: "+ " " + male_df["pref_maslak_sect"].astype(str) + " \n" +
    #     "Siblings: "+ " " + male_df["pref_no_of_siblings"].astype(str) + " \n" +
    #     "Occupation: "+ " " + male_df["pref_work_job"].astype(str) + " \n" +
    #     "Go to dargah: "+ " " + male_df["pref_go_to_dargah"].astype(str) + " \n" +
    #     "Mother tongue: "+ " " + male_df["pref_mother_tongue"].astype(str) + " \n" +
    #     "Deender: "+ " " + male_df["pref_deendari"].astype(str) + " \n" +
    #     "location: "+ " " + male_df["pref_location"].astype(str) + " \n" +
    #     "pref_own_house: "+ " " + male_df["pref_own_house"].astype(str) + " \n" +
    #     "Preferences: "+ " " + male_df["preferences"].astype(str)
    # )
    # female_df["text"] = (
    #     female_df["profile_id"].astype(str) + " \n" +
    #     female_df["full_name"].astype(str) + " \n" +
    #     "Date Of Birth: "+ " " + female_df["date_of_birth"].astype(str) + " \n" +
    #     "Age: "+ " " + female_df["age"].astype(str) + " \n" +
    #     "Marital Status: "+ " " + female_df["marital_status"].astype(str) + " \n" +
    #     "Gender: "+ " " + female_df["gender"].astype(str) + " \n" +
    #     "complexion: "+ " " + female_df["complexion"].astype(str) + " \n" +
    #     "Education: "+ " " + female_df["education"].astype(str) + " \n" +
    #     "Height: "+ " " + female_df["height"].astype(str) + " \n" +
    #     "Native_place: "+ " " + female_df["native_place"].astype(str) + " \n" +
    #     "residence: "+ " " + female_df["residence"].astype(str) + " \n" +
    #     "Father: "+ " " + female_df["father"].astype(str) + " \n" +
    #     "Mother: "+ " " + female_df["mother"].astype(str) + " \n" +
    #     "sect: "+ " " + female_df["sect"].astype(str) + " \n" +
    #     "religious_practice,: "+ " " + female_df["religious_practice"].astype(str) + " \n" +
    #     "go_to_dargah: "+ " " + female_df["go_to_dargah"].astype(str) + " \n" +
    #     "occupation: "+ " " + female_df["occupation"].astype(str) + " \n" +
    #     "age_range: "+ " " + female_df["pref_age_range"].astype(str) + " \n" +
    #     "Marital Status: "+ " " + female_df["pref_marital_status"].astype(str) + " \n" +
    #     "Complexion: "+ " " + female_df["pref_complexion"].astype(str) + " \n" +
    #     "Education: "+ " " + female_df["pref_education"].astype(str) + " \n" +
    #     "Height: "+ " " + female_df["pref_height"].astype(str) + " \n" +
    #     "Native_place: "+ " " + female_df["pref_native_place"].astype(str) + " \n" +
    #     "Maslak_sect: "+ " " + female_df["pref_maslak_sect"].astype(str) + " \n" +
    #     "Siblings: "+ " " + female_df["pref_no_of_siblings"].astype(str) + " \n" +
    #     "Occupation: "+ " " + female_df["pref_work_job"].astype(str) + " \n" +
    #     "Go to dargah: "+ " " + female_df["pref_go_to_dargah"].astype(str) + " \n" +
    #     "Mother tongue: "+ " " + female_df["pref_mother_tongue"].astype(str) + " \n" +
    #     "Deender: "+ " " + female_df["pref_deendari"].astype(str) + " \n" +
    #     "location: "+ " " + female_df["pref_location"].astype(str) + " \n" +
    #     "pref_own_house: "+ " " + female_df["pref_own_house"].astype(str) + " \n" +
    #     "Preferences: "+ " " + female_df["preferences"].astype(str)
        
    # )
    # return female_df["text"].tolist()
    male_embeddings = model.encode(male_df["text"].tolist()).astype("float32")
    female_embeddings = model.encode(female_df["text"].tolist()).astype("float32")
    
    # #     # Normalize embeddings
    male_embeddings /= np.linalg.norm(male_embeddings, axis=1, keepdims=True)
    female_embeddings /= np.linalg.norm(female_embeddings, axis=1, keepdims=True)

    # # Create FAISS indexes **SEPARATELY**
    male_index = faiss.IndexFlatL2(male_embeddings.shape[1])
    female_index = faiss.IndexFlatL2(female_embeddings.shape[1])

    male_index.add(male_embeddings)  # type: ignore # Male Index (to match Females)
    female_index.add(female_embeddings)  # type: ignore # Female Index (to match Males)

    os.makedirs("store", exist_ok=True)

    # # Save FAISS indexes separately
    faiss.write_index(male_index, "store/male_index.faiss")   # Male users will search here (matches Female)
    faiss.write_index(female_index, "store/female_index.faiss")  # Female users will search here (matches Male)

    print("✅ FAISS indexes created successfully.")
    
# texts = create_faiss_index()
# print(texts[0])



