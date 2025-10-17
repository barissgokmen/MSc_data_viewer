import json

def read_keyboard_data(filename='keyboard.json'):
    """Read and parse keyboard data from JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {filename}")
        return []
    

def segment_messages(data):
   
    """Segment messages from keyboard data."""
    if not data:
        return []
        
    segments = []
    current_segment = []

    
    for entry in data[1:]:
        current_text = entry.get('current_text', '')[1:-1]  # Remove brackets
        before_text = entry.get('before_text', '')  # No brackets in before_text

        if len(current_text) == 0 or len(before_text) == 0:
            segments.append(current_segment)
            current_segment = []  # Reset current segment
            continue

        # Check if current text and before text start with same character
        if current_text[0] == before_text[0]:
            current_segment.append(entry)
        else:
            # If they don't match, finalize the current segment
            segments.append(current_segment)
            current_segment = [entry]  # Start new segment with current entry
        
        previous_text = current_text
    
    if current_segment:
        segments.append(current_segment)


    # Remove empty segments
    segments = [segment for segment in segments if segment]
    # Remove segments with less than 4 entries
    segments = [segment for segment in segments if len(segment) >= 4]
    
    return segments

def print_segment(segment):
    """ take the last entry of the segment and print all it in a readable format """
    
    if segment:
        for entry in segment:
            print(f"ID: {entry['_id']}, Text: {entry['current_text']}, App: {entry['package_name']}")

def main():
    # Read keyboard data
    keyboard_data = read_keyboard_data()
    
    if keyboard_data:
        print(f"Loaded {len(keyboard_data)} keyboard entries")
        
        for entry in keyboard_data:
            continue
            print(f"ID: {entry['_id']}, Text: {entry['current_text']}, App: {entry['package_name']}")
    else:
        print("No data loaded")

    # Segment messages
    segments = segment_messages(keyboard_data)
    print(f"Segmented into {len(segments)} messages")

    # Print each segment
    for i, segment in enumerate(segments):
        print(f"\nSegment {i + 1}:")
        print_segment(segment)

        # Print the last entry of the segment
        if segment:
            last_entry = segment[-1]
            print(f"Last Entry ID: {last_entry['_id']}, Text: {last_entry['current_text']}, App: {last_entry['package_name']}")


    



if __name__ == "__main__":
    main()