


import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
import gdown # <-- Import gdown
import os # <-- Import os to check if file exists

# --- Constants ---
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_K = 30
# PKL_FILE = "paragraphs_with_embeddings_v2.pkl" # <-- Comment out or remove old PKL_FILE
# --- IMPORTANT: Replace 'YOUR_GOOGLE_DRIVE_FILE_ID' with the actual ID of your PKL file ---
# Or you can use the full shareable URL if you prefer, gdown can often handle it.
# Example using File ID:
GOOGLE_DRIVE_FILE_ID = "1-d46DU4g3LpWUl3B0zgCkvWa9pYkeQe4" # e.g., "123AbcXYZ789..."
# Example using full URL (make sure it's a direct download link or shareable link):
# GOOGLE_DRIVE_FILE_URL = "https://drive.google.com/file/d/1-d46DU4g3LpWUl3B0zgCkvWa9pYkeQe4/view?usp=drive_link"
LOCAL_PKL_FILE_NAME = "downloaded_paragraphs_with_embeddings_v2.pkl" # Local name for the downloaded file

SIMILARITY_THRESHOLD = 0.4


# --- UI Config (Optional - you had this commented out, uncomment if needed) ---
# st.title("🔍 Multilingual Paragraph Search")
# st.subheader("Built for PDF content in 2-column layout")
# query = st.text_input("Enter a detailed query (minimum 5 words):")
# threshold = st.slider(
#     "Similarity Threshold (lower = more results, higher = more relevant)",
#     min_value=0.1,
#     max_value=0.9,
#     value=0.4,
#     step=0.05
# )


# --- Init ---
@st.cache_resource
def load_model():
    return SentenceTransformer(MODEL_NAME)

@st.cache_data
def load_dataset():
    # --- Download from Google Drive if not already downloaded ---
    if not os.path.exists(LOCAL_PKL_FILE_NAME):
        st.info(f"Downloading dataset from Google Drive (this may take a moment)...")
        try:
            # Using file ID:
            gdown.download(id=GOOGLE_DRIVE_FILE_ID, output=LOCAL_PKL_FILE_NAME, quiet=False)
            # Or using URL:
            # gdown.download(url=GOOGLE_DRIVE_FILE_URL, output=LOCAL_PKL_FILE_NAME, quiet=False)
            st.success("Dataset downloaded successfully!")
        except Exception as e:
            st.error(f"Error downloading file from Google Drive: {e}")
            st.error("Please ensure the Google Drive File ID is correct and the file is shared with 'Anyone with the link'.")
            return pd.DataFrame() # Return empty DataFrame on error
    else:
        st.info("Dataset already downloaded. Loading from local cache.")

    try:
        return pd.read_pickle(LOCAL_PKL_FILE_NAME)
    except Exception as e:
        st.error(f"Error reading the PKL file: {e}")
        return pd.DataFrame() # Return empty DataFrame on error


model = load_model()
translator = Translator()
df = load_dataset()

# --- App UI ---
st.title(" 🌐 Multilingual Paragraph Search")
st.markdown("📄 Search through multilingual technical documents (belonging to 'ZS115670_', '47797881_ series'), having 2 column layout.")

query = st.text_input("Enter search query (at least 6 words):")
# translate_toggle = st.checkbox("🔁 Include translated paragraph (optional)", value=False)

if st.button("🔍 Search"):
    if df.empty: # Check if DataFrame is empty (e.g., due to download error)
        st.error("Dataset could not be loaded. Please check the logs or try again.")
    elif not query or len(query.strip().split()) < 6:
        st.warning("Please enter a search query with more than 5 words.")
    else:
        query_embedding = model.encode(query)
        # Ensure 'embedding' column exists and is not empty before proceeding
        if 'embedding' not in df.columns or df['embedding'].isnull().all():
            st.error("The 'embedding' column is missing or empty in the dataset.")
        else:
            try:
                para_embeddings = np.vstack(df['embedding'].dropna().to_numpy()) # Add .dropna() just in case
                if para_embeddings.ndim == 1: # Handle case where only one valid embedding exists
                    para_embeddings = para_embeddings.reshape(1, -1)

                if para_embeddings.shape[0] == 0: # No valid embeddings found
                    st.info("No embeddings available in the dataset to compare with.")
                else:
                    scores = cosine_similarity([query_embedding], para_embeddings)[0]

                    df_temp = df.dropna(subset=['embedding']).copy() # Work on a copy with no NaN embeddings
                    # Ensure scores align with df_temp if there were NaNs
                    if len(scores) == len(df_temp):
                        df_temp['score'] = scores
                        df_filtered = df_temp[df_temp['score'] >= SIMILARITY_THRESHOLD]
                        df_top = df_filtered.sort_values(by="score", ascending=False).head(TOP_K).copy()

                        if df_top.empty:
                            st.info("No matching paragraphs found for the given threshold.")
                        else:
                            st.success(f"Found {len(df_top)} matching paragraphs.")

                            # Display table
                            display_cols = ["doc_name", "page_number", "paragraph", "language", "score"]
                            st.dataframe(df_top[display_cols], use_container_width=True)

                            # Download
                            def convert_df_to_csv(df_export):
                                return df_export.to_csv(index=False, encoding="utf-8-sig")

                            csv = convert_df_to_csv(df_top[display_cols])
                            st.download_button(
                                label="📥 Download results as CSV",
                                data=csv,
                                file_name="filtered_paragraphs.csv",
                                mime="text/csv"
                            )
                    else:
                        st.error("Mismatch between scores and dataframe rows. This might be due to NaN embeddings.")
            except Exception as e:
                st.error(f"An error occurred during search: {e}")









###################################





















# import streamlit as st
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from googletrans import Translator

# # --- Constants ---
# MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# TOP_K = 30
# PKL_FILE = "paragraphs_with_embeddings_v2.pkl"
# SIMILARITY_THRESHOLD = 0.4


# # # --- UI Config ---
# # st.title("🔍 Multilingual Paragraph Search")
# # st.subheader("Built for PDF content in 2-column layout")

# # query = st.text_input("Enter a detailed query (minimum 5 words):")

# # # 👇 Add this slider to let user choose similarity threshold
# # threshold = st.slider(
# #     "Similarity Threshold (lower = more results, higher = more relevant)", 
# #     min_value=0.1, 
# #     max_value=0.9, 
# #     value=0.4, 
# #     step=0.05
# # )



# # --- Init ---
# @st.cache_resource
# def load_model():
#     return SentenceTransformer(MODEL_NAME)

# @st.cache_data
# def load_dataset():
#     return pd.read_pickle(PKL_FILE)

# model = load_model()
# translator = Translator()
# df = load_dataset()

# # --- App UI ---
# st.title(" 🌐 Multilingual Paragraph Search")
# st.markdown("📄 Search through multilingual technical documents (belonging to 'ZS115670_', '47797881_ series'), having 2 column layout.")

# query = st.text_input("Enter search query (at least 6 words):")
# # translate_toggle = st.checkbox("🔁 Include translated paragraph (optional)", value=False)

# if st.button("🔍 Search"):

#     if not query or len(query.strip().split()) < 6:
#         st.warning("Please enter a search query with more than 5 words.")
#     else:
#         query_embedding = model.encode(query)
#         para_embeddings = np.vstack(df['embedding'].to_numpy())
#         scores = cosine_similarity([query_embedding], para_embeddings)[0]

#         df['score'] = scores
#         df_filtered = df[df['score'] >= SIMILARITY_THRESHOLD]
#         df_top = df_filtered.sort_values(by="score", ascending=False).head(TOP_K).copy()

#         if df_top.empty:
#             st.info("No matching paragraphs found.")
#         else:
#             st.success(f"Found {len(df_top)} matching paragraphs.")
            
#             # Optional Translation
#             # if translate_toggle:
#             #     df_top["translated_paragraph"] = df_top["paragraph"].apply(
#             #         lambda x: translator.translate(x, dest="en").text
#             #     )

#             # Display table
#             display_cols = ["doc_name", "page_number", "paragraph", "language", "score"]
#             # if translate_toggle:
#             #     display_cols.append("translated_paragraph")
#             st.dataframe(df_top[display_cols], use_container_width=True)

#             # Download
#             def convert_df_to_csv(df_export):
#                 return df_export.to_csv(index=False, encoding="utf-8-sig")

#             csv = convert_df_to_csv(df_top[display_cols])
#             st.download_button(
#                 label="📥 Download results as CSV",
#                 data=csv,
#                 file_name="filtered_paragraphs.csv",
#                 mime="text/csv"
#             )


#######################################################
#changed the script to pull model from another repo below 
#####################################################


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# from googletrans import Translator
# import os

# # --- Constants ---
# MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# TOP_K = 30
# PKL_FILE = "paragraphs_with_embeddings_v2.pkl"  # Path to the .pkl file in the repo
# SIMILARITY_THRESHOLD = 0.4

# # --- Cached loaders ---
# @st.cache_resource
# def load_model():
#     return SentenceTransformer(MODEL_NAME)

# @st.cache_data
# def load_dataset():
#     # Check if the .pkl file exists in the repo directory
#     if not os.path.exists(PKL_FILE):
#         st.error(f"Error: The file {PKL_FILE} was not found in the repository.")
#         raise FileNotFoundError(f"{PKL_FILE} not found.")
    
#     # Load the dataset from the local .pkl file
#     return pd.read_pickle(PKL_FILE)

# # --- Load resources ---
# model = load_model()
# translator = Translator()

# # Load the dataset
# try:
#     df = load_dataset()
# except FileNotFoundError:
#     st.error(f"Could not find the file {PKL_FILE}. Please check the file path and try again.")
#     st.stop()

# # --- UI ---
# st.title("🌐 Multilingual Paragraph Search")
# st.markdown("Search technical documents with multilingual paragraphs (PDFs in 2-column layout).")

# query = st.text_input("Enter a detailed search query (minimum 6 words):")
# # Uncomment if translation is needed
# # translate_toggle = st.checkbox("Include translated paragraph", value=False)

# if st.button("🔍 Search"):
#     if not query or len(query.strip().split()) < 6:
#         st.warning("Please enter a query with more than 5 words.")
#     else:
#         query_embedding = model.encode(query)
#         para_embeddings = np.vstack(df['embedding'].to_numpy())
#         scores = cosine_similarity([query_embedding], para_embeddings)[0]
#         df['score'] = scores

#         df_filtered = df[df['score'] >= SIMILARITY_THRESHOLD]
#         df_top = df_filtered.sort_values(by="score", ascending=False).head(TOP_K).copy()

#         if df_top.empty:
#             st.info("No matching paragraphs found.")
#         else:
#             st.success(f"Found {len(df_top)} matching paragraphs.")

#             # Uncomment to translate
#             # if translate_toggle:
#             #     df_top["translated_paragraph"] = df_top["paragraph"].apply(
#             #         lambda x: translator.translate(x, dest="en").text
#             #     )

#             display_cols = ["doc_name", "page_number", "paragraph", "language", "score"]
#             # if translate_toggle:
#             #     display_cols.append("translated_paragraph")

#             st.dataframe(df_top[display_cols], use_container_width=True)

#             def convert_df_to_csv(df_export):
#                 return df_export.to_csv(index=False, encoding="utf-8-sig")

#             csv = convert_df_to_csv(df_top[display_cols])

#             st.download_button(
#                 label="⬇️ Download results as CSV",
#                 data=csv,
#                 file_name="filtered_paragraphs.csv",
#                 mime="text/csv"
#             )


