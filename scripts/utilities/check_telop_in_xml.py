"""
Check if telop information exists in XML files
"""
from fcpxml_to_tracks import FCPXMLParser

parser = FCPXMLParser()
tracks = parser.parse_fcpxml('editxml/bandicam 2025-03-03 22-34-57-492.xml')

print(f"Total tracks: {len(tracks)}")
print(f"Track type: {type(tracks)}")

if len(tracks) > 0:
    print(f"\nFirst track type: {type(tracks[0])}")
    print(f"First track length: {len(tracks[0]) if isinstance(tracks[0], list) else 'N/A'}")
    
    # tracks is a list of lists (one list per track)
    for track_idx, track_data in enumerate(tracks[:5]):
        print(f"\n=== Track {track_idx} ===")
        print(f"  Clips in track: {len(track_data)}")
        
        # Check each clip in the track
        for clip_idx, clip in enumerate(track_data[:3]):  # First 3 clips
            if isinstance(clip, dict):
                text = clip.get('graphics_text', '')
                if text:
                    print(f"  Clip {clip_idx}:")
                    print(f"    Start: {clip.get('start', 0):.2f}s")
                    print(f"    Duration: {clip.get('duration', 0):.2f}s")
                    print(f"    Text: {text}")
                    print(f"    Clip name: {clip.get('clip_name', '')}")
