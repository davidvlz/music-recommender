"""
Utility functions for the Spotify Music Recommendation System.

This module provides helper functions used throughout the application.
"""

import difflib
import numpy as np

# Import configuration
import config


def find_closest_songs(user_input, song_data, artist=None, year=None, limit=None):
    """
    Find songs with names closest to user input.

    Args:
        user_input (str): User input song name
        song_data (pandas.DataFrame): DataFrame containing song data
        artist (str, optional): Artist name for better matching
        year (int, optional): Year of release for better matching
        limit (int, optional): Maximum number of matches to return

    Returns:
        list: List of dictionaries with matching song information
    """
    limit = limit or config.MAX_MATCHES_TO_SHOW

    # Convert user input to lowercase for case-insensitive matching
    user_input_lower = user_input.lower()

    # Get all song names from the dataset
    song_names = song_data['name'].tolist()
    song_names_lower = [name.lower() for name in song_names]

    # Find closest matches using difflib
    closest_matches = difflib.get_close_matches(
        user_input_lower,
        list(set(song_names_lower)),  # Use unique song names for matching
        n=limit * 3,  # Get more matches initially to allow for filtering
        cutoff=config.FUZZY_MATCH_CUTOFF
    )

    # Find all indices for each match, to handle duplicates
    matched_indices = []
    for match in closest_matches:
        # Find all occurrences of this match
        indices = [i for i, name in enumerate(song_names_lower) if name == match]
        matched_indices.extend(indices)

    # Filter matches by artist and year if provided
    results = []
    seen_combinations = set()  # To track unique song-artist combinations

    for idx in matched_indices:
        song_row = song_data.iloc[idx]

        # Create a key to check for duplicates
        song_artist_key = (song_row['name'].lower(), song_row['artists'].lower())

        # Skip if we've already seen this exact song-artist combination
        if song_artist_key in seen_combinations:
            continue

        seen_combinations.add(song_artist_key)

        # Check if artist matches (if provided)
        artist_match = True
        if artist:
            artist_match = artist.lower() in song_row['artists'].lower()

        # Check if year matches (if provided)
        year_match = True
        if year and 'year' in song_row:
            year_match = int(song_row['year']) == year

        # Add to results if all criteria match
        if artist_match and year_match:
            results.append({
                'name': song_row['name'],
                'artists': song_row['artists'],
                'year': song_row['year'] if 'year' in song_row else None,
                'popularity': song_row['popularity'] if 'popularity' in song_row else None,
                'index': idx
            })

        # Stop once we have enough results
        if len(results) >= limit:
            break

    return results


def prompt_for_song_selection(matches):
    """
    Prompt the user to select a song from a list of matches.
    
    Args:
        matches (list): List of dictionaries with matching song information
        
    Returns:
        dict or None: Selected song information or None if no selection was made
    """
    # TODO: Implement song selection prompt
    if not matches:
        print("No matches found.")
        return None
    
    print("\nDid you mean:")
    for i, match in enumerate(matches):
        # Format the match
        match_info = f"{i+1}. {match['name']} by {match['artists']}"
        
        # Add year if available
        if match['year']:
            match_info += f" ({match['year']})"
        
        print(match_info)
    
    print(f"{len(matches)+1}. None of these")
    
    # Get user selection
    while True:
        try:
            choice = input("\nYour choice: ")
            choice = int(choice)
            
            if choice < 1 or choice > len(matches) + 1:
                print(f"Please enter a number between 1 and {len(matches)+1}.")
                continue
            
            if choice == len(matches) + 1:
                return None
            
            return matches[choice - 1]
        
        except ValueError:
            print("Please enter a valid number.")
        except Exception as e:
            print(f"Error: {e}")
            return None


def calculate_similarity(distance):
    """
    Calculate similarity score from distance.
    
    Args:
        distance (float): Distance between two songs
        
    Returns:
        float: Similarity score (0.0 to 1.0)
    """
    # TODO: Implement similarity calculation
    # Convert distance to similarity (1.0 for identical songs, approaching 0.0 for very different songs)
    return 1.0 / (1.0 + distance)


def get_popular_songs(processed_df, n=10):
    """
    Get a list of popular songs from the dataset.
    
    Args:
        processed_df (pandas.DataFrame): Preprocessed DataFrame
        n (int, optional): Number of songs to return
        
    Returns:
        list: List of dictionaries with popular song information
    """
    # TODO: Implement getting popular songs
    # Check if popularity column exists
    if 'popularity' not in processed_df.columns:
        # If no popularity column, return random songs
        random_indices = np.random.choice(len(processed_df), size=n, replace=False)
        popular_songs = []
        
        for idx in random_indices:
            song = processed_df.iloc[idx]
            popular_songs.append({
                'name': song['name'],
                'artists': song['artists'],
                'year': song['year'] if 'year' in song else None,
                'index': idx
            })
        
        return popular_songs
    
    # Sort by popularity and get top N
    top_songs = processed_df.sort_values('popularity', ascending=False).head(n)
    
    popular_songs = []
    for idx, song in top_songs.iterrows():
        popular_songs.append({
            'name': song['name'],
            'artists': song['artists'],
            'year': song['year'] if 'year' in song else None,
            'popularity': song['popularity'],
            'index': idx
        })
    
    return popular_songs


def print_song_list(songs, title="Songs"):
    """
    Print a formatted list of songs.
    
    Args:
        songs (list): List of dictionaries with song information
        title (str, optional): Title for the list
        
    Returns:
        None
    """
    # TODO: Implement printing song list
    print(f"\n{title}:")
    
    for i, song in enumerate(songs):
        # Format the song
        song_info = f"{i+1}. {song['name']} by {song['artists']}"
        
        # Add year if available
        if song['year']:
            song_info += f" ({song['year']})"
        
        # Add popularity if available
        if 'popularity' in song and song['popularity']:
            song_info += f" - Popularity: {song['popularity']}"
        
        print(song_info)


def clear_screen():
    """
    Clear the terminal screen.
    
    Returns:
        None
    """
    # TODO: Implement clearing screen
    import os
    
    # Clear screen command based on OS
    if os.name == 'nt':  # Windows
        os.system('cls')
    else:  # Unix/Linux/MacOS
        os.system('clear')


if __name__ == "__main__":
    # Example usage
    from data_loader import get_data
    
    # Get the data
    processed_df, _, _ = get_data()
    
    if processed_df is not None:
        # Test fuzzy matching
        test_song = "highway to well"
        matches = find_closest_songs(test_song, processed_df)
        
        print(f"Matches for '{test_song}':")
        for i, match in enumerate(matches):
            print(f"{i+1}. {match['name']} by {match['artists']}")
        
        # Test getting popular songs
        popular_songs = get_popular_songs(processed_df)
        
        print_song_list(popular_songs, title="Popular Songs")
