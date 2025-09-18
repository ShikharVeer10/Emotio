from keybert import KeyBERT

kw_model = KeyBERT()

def extract_keywords(text, sentiment_label, top_n=5):
    """
    Extract keywords from the text using KeyBERT.
    The sentiment_label can be used to customize keyword extraction if needed.
    """
    keywords = kw_model.extract_keywords(text, top_n=top_n)
    # Return only the keywords (without scores) as a list
    return [kw[0] for kw in keywords]
