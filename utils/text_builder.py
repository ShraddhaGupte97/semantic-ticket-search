def build_ticket_text(row):
    return (
        f"Ticket Summary: {row['title']}. "
        f"Focus Metric: {row['metrics']}. "
        f"Breakdowns: {', '.join(row['dimensions'])}. "
        f"Folder: {row['assigned_folder']}. Label: {row['labels']}. Time: {row['time_period']}."
    )