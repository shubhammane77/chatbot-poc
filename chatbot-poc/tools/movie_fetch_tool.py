import requests

def fetch_movie_by_title_year(title: str, year: int, api_key: str = "f28fe471"):
    """
    Fetches movie information from the OMDb API by movie title and year.
    This tool can be used by AI to retrieve movie information.
    Args:
        title (str): The movie title to search for.
        year (int): The release year of the movie.
        api_key (str): OMDb API key.
    Returns:
        dict: Movie information as returned by OMDb API.
    """
    base_url = "http://www.omdbapi.com/"
    params = {
        "t": title,
        "y": year,
        "r": "json",
        "apikey": api_key
    }
    print(f"Fetching movie data for title: {title} and year: {year}")
    response = requests.get(base_url, params=params)

    response.raise_for_status()
    return response.json()