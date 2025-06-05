import os
from utils.io import load_dataframe
from utils.text_builder import build_ticket_text
from utils.helpers import has_similar_ticket

def prepare_ticket_data(json_path = "data/raw_data.json"):
    """
    Loads ticket data, filters rows with similar tickets,
    and constructs the 'embedding_input' column.
    
    Returns:
        pd.DataFrame: processed dataframe
    """
    df = load_dataframe(json_path)
    df_train = df[df.apply(has_similar_ticket, axis=1)].copy()
    df_train["embedding_input"] = df_train.apply(build_ticket_text, axis=1)
    return df_train