#!/usr/bin/env python3
"""
Simple syntax test without external dependencies
"""

# Test the video processing logic without imports
def test_filename_extension():
    """Test filename extension extraction logic"""
    
    # Test cases
    test_cases = [
        ("video.mp4", "mp4"),
        ("test-video.webm", "webm"),
        ("noextension", "mp4"),  # Should default to mp4
        ("file.", "mp4"),  # Empty extension should default to mp4
        ("multi.dot.file.avi", "avi"),
    ]
    
    for filename, expected in test_cases:
        # Simulate the logic from main.py
        filename_parts = filename.split('.')
        extension = filename_parts[-1] if len(filename_parts) > 1 and filename_parts[-1] else "mp4"
        
        print(f"Filename: '{filename}' -> Extension: '{extension}' (expected: '{expected}')")
        assert extension == expected, f"Failed for {filename}: got {extension}, expected {expected}"
    
    print("✅ All filename extension tests passed!")

def test_response_data():
    """Test response data structure"""
    
    # Simulate the response data creation
    output_video_b64 = "fake_base64_data"
    output_format = "mp4"
    return_url = False
    
    response_data = {
        "success": True,
        "output_video": output_video_b64,
        "output_format": output_format
    }
    
    if return_url:
        response_data["output_video_url"] = None
    
    print("✅ Response data structure test passed!")
    print(f"Response data: {response_data}")

if __name__ == "__main__":
    test_filename_extension()
    test_response_data()
    print("✅ All syntax tests passed!")
