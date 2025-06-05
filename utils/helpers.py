def has_similar_ticket(row):
    try:
        return "Similar ticket" in row["comments"][0]["comment"]
    except:
        return False