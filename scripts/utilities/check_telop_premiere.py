"""
Check telop information in Premiere XML
"""
from premiere_xml_parser import PremiereXMLParser

parser = PremiereXMLParser()
result = parser.parse_xml('editxml/bandicam 2025-03-03 22-34-57-492.xml')

print(f"Parse result type: {type(result)}")
print(f"Result keys: {result.keys() if isinstance(result, dict) else 'N/A'}")

if isinstance(result, dict) and 'clips' in result:
    clips = result['clips']
    print(f"\nTotal clips: {len(clips)}")
    
    # Check for telop/text
    has_text = False
    for i, clip in enumerate(clips[:20]):
        text = clip.get('text_content', '') or clip.get('graphics_text', '')
        if text:
            has_text = True
            print(f"\nClip {i}:")
            print(f"  Start: {clip.get('start', 0):.2f}s")
            print(f"  Duration: {clip.get('duration', 0):.2f}s")
            print(f"  Text: {text}")
            print(f"  Name: {clip.get('name', '')}")
    
    if not has_text:
        print("\n⚠️ No telop/text found in first 20 clips")
        print("\nSample clip structure:")
        if clips:
            print(clips[0])
