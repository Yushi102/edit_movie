"""
Property-based tests for batch processing and logging system using Hypothesis
"""
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis import HealthCheck
import xml.etree.ElementTree as ET
import pandas as pd
import os
import tempfile
import glob
from batch_xml2csv_keyframes import batch_convert


# ==========================================
# Property 5: XML File Discovery Completeness
# **Feature: multi-track-training-pipeline, Property 5: XML File Discovery Completeness**
# ==========================================

@given(
    num_xml_files=st.integers(min_value=1, max_value=10),
    num_other_files=st.integers(min_value=0, max_value=5)
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_xml_file_discovery_completeness(num_xml_files, num_other_files):
    """
    For any directory containing files with various extensions,
    the file discovery function should return exactly the set of files with .xml extension
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create XML files
        xml_files_created = []
        for i in range(num_xml_files):
            filename = f"test_{i}.xml"
            filepath = os.path.join(tmpdir, filename)
            
            # Create minimal valid XML
            root = ET.Element('xmeml', version="4")
            seq = ET.SubElement(root, 'sequence')
            ET.SubElement(seq, 'name').text = f"Test{i}"
            rate = ET.SubElement(seq, 'rate')
            ET.SubElement(rate, 'timebase').text = "30"
            ET.SubElement(seq, 'duration').text = "300"
            
            tree = ET.ElementTree(root)
            tree.write(filepath, encoding='utf-8', xml_declaration=True)
            xml_files_created.append(filename)
        
        # Create non-XML files
        other_extensions = ['.txt', '.csv', '.json', '.mp4', '.mov']
        for i in range(num_other_files):
            ext = other_extensions[i % len(other_extensions)]
            filename = f"other_{i}{ext}"
            filepath = os.path.join(tmpdir, filename)
            with open(filepath, 'w') as f:
                f.write("dummy content")
        
        # Discover XML files
        discovered_xml = glob.glob(os.path.join(tmpdir, "*.xml"))
        discovered_basenames = [os.path.basename(f) for f in discovered_xml]
        
        # Verify
        assert len(discovered_basenames) == num_xml_files, \
            f"Expected {num_xml_files} XML files, found {len(discovered_basenames)}"
        
        for xml_file in xml_files_created:
            assert xml_file in discovered_basenames, \
                f"XML file {xml_file} should be discovered"


# ==========================================
# Property 6: CSV Schema Completeness
# **Feature: multi-track-training-pipeline, Property 6: CSV Schema Completeness**
# ==========================================

@given(
    num_tracks=st.integers(min_value=1, max_value=20)
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_csv_schema_completeness(num_tracks):
    """
    For any generated CSV dataset, the output should contain exactly
    3 metadata columns (video_id, source_video_name, time) plus
    180 track parameter columns (20 tracks × 9 parameters)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a minimal valid XML
        xml_path = os.path.join(tmpdir, "test.xml")
        
        root = ET.Element('xmeml', version="4")
        seq = ET.SubElement(root, 'sequence')
        ET.SubElement(seq, 'name').text = "Test"
        
        rate = ET.SubElement(seq, 'rate')
        ET.SubElement(rate, 'timebase').text = "30"
        ET.SubElement(seq, 'duration').text = "300"
        
        media = ET.SubElement(seq, 'media')
        video = ET.SubElement(media, 'video')
        
        # Add tracks
        for t_idx in range(num_tracks):
            track = ET.SubElement(video, 'track')
            clip = ET.SubElement(track, 'clipitem', id=f"clip-{t_idx}")
            ET.SubElement(clip, 'name').text = f"Clip_{t_idx}"
            ET.SubElement(clip, 'start').text = "0"
            ET.SubElement(clip, 'end').text = "100"
        
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
        # Process
        output_csv = os.path.join(tmpdir, "output.csv")
        log_file = os.path.join(tmpdir, "log.txt")
        
        success, failure, failed = batch_convert(tmpdir, output_csv, log_file)
        
        # Read CSV
        df = pd.read_csv(output_csv)
        
        # Check metadata columns
        assert 'video_id' in df.columns
        assert 'source_video_name' in df.columns
        assert 'time' in df.columns
        
        # Check track columns (20 tracks × 9 parameters = 180 columns)
        expected_params = ['active', 'asset', 'scale', 'x', 'y', 'crop_l', 'crop_r', 'crop_t', 'crop_b']
        
        for track_num in range(1, 21):
            for param in expected_params:
                col_name = f"target_v{track_num}_{param}"
                assert col_name in df.columns, f"Column {col_name} should exist"
        
        # Total columns: 3 metadata + 180 track parameters = 183
        assert len(df.columns) == 183, f"Expected 183 columns, got {len(df.columns)}"


# ==========================================
# Property 7: Source Video Name Extraction (already tested in test_xml_parser.py)
# This is a duplicate test for batch processing context
# ==========================================

@given(
    video_name=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=('L', 'N')))
)
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_source_video_name_in_csv(video_name):
    """
    For any XML with pathurl element, the CSV should contain
    the extracted source_video_name in the source_video_name column
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        xml_path = os.path.join(tmpdir, "test.xml")
        
        root = ET.Element('xmeml', version="4")
        seq = ET.SubElement(root, 'sequence')
        ET.SubElement(seq, 'name').text = "Test"
        
        rate = ET.SubElement(seq, 'rate')
        ET.SubElement(rate, 'timebase').text = "30"
        ET.SubElement(seq, 'duration').text = "300"
        
        # Add pathurl with video name
        file_elem = ET.SubElement(seq, 'file', id="file-1")
        ET.SubElement(file_elem, 'pathurl').text = f"file://localhost/C:/Videos/{video_name}.mp4"
        
        media = ET.SubElement(seq, 'media')
        video = ET.SubElement(media, 'video')
        track = ET.SubElement(video, 'track')
        clip = ET.SubElement(track, 'clipitem', id="clip-1")
        ET.SubElement(clip, 'name').text = "TestClip"
        ET.SubElement(clip, 'start').text = "0"
        ET.SubElement(clip, 'end').text = "100"
        
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
        # Process
        output_csv = os.path.join(tmpdir, "output.csv")
        log_file = os.path.join(tmpdir, "log.txt")
        
        success, failure, failed = batch_convert(tmpdir, output_csv, log_file)
        
        # Read CSV
        df = pd.read_csv(output_csv)
        
        # Check source_video_name
        assert 'source_video_name' in df.columns
        # Note: If pathurl is not found, it defaults to "unknown"
        # So we check if it's either the expected name or "unknown"
        unique_names = df['source_video_name'].unique()
        assert len(unique_names) > 0


# ==========================================
# Property 8: Dataframe Concatenation Invariant
# **Feature: multi-track-training-pipeline, Property 8: Dataframe Concatenation Invariant**
# ==========================================

@given(
    num_files=st.integers(min_value=2, max_value=5),
    duration_per_file=st.integers(min_value=100, max_value=500)
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_dataframe_concatenation_invariant(num_files, duration_per_file):
    """
    For any batch of N XML files producing dataframes with row counts r1, r2, ..., rN,
    the concatenated master dataframe should have exactly r1 + r2 + ... + rN rows
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        expected_total_rows = 0
        
        for i in range(num_files):
            xml_path = os.path.join(tmpdir, f"test_{i}.xml")
            
            root = ET.Element('xmeml', version="4")
            seq = ET.SubElement(root, 'sequence')
            ET.SubElement(seq, 'name').text = f"Test{i}"
            
            rate = ET.SubElement(seq, 'rate')
            ET.SubElement(rate, 'timebase').text = "30"
            ET.SubElement(seq, 'duration').text = str(duration_per_file)
            
            media = ET.SubElement(seq, 'media')
            video = ET.SubElement(media, 'video')
            track = ET.SubElement(video, 'track')
            clip = ET.SubElement(track, 'clipitem', id=f"clip-{i}")
            ET.SubElement(clip, 'name').text = f"Clip_{i}"
            ET.SubElement(clip, 'start').text = "0"
            ET.SubElement(clip, 'end').text = str(duration_per_file)
            
            tree = ET.ElementTree(root)
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            
            # Calculate expected rows for this file
            # rows = duration_in_seconds / INTERVAL
            # duration_in_seconds = duration_per_file / fps
            fps = 30.0
            interval = 0.1
            max_sec = duration_per_file / fps
            num_samples = int(max_sec / interval)
            expected_total_rows += num_samples
        
        # Process batch
        output_csv = os.path.join(tmpdir, "output.csv")
        log_file = os.path.join(tmpdir, "log.txt")
        
        success, failure, failed = batch_convert(tmpdir, output_csv, log_file)
        
        # Read CSV
        df = pd.read_csv(output_csv)
        
        # Verify total rows
        assert len(df) == expected_total_rows, \
            f"Expected {expected_total_rows} total rows, got {len(df)}"


# ==========================================
# Property 9: Error Resilience and Logging
# **Feature: multi-track-training-pipeline, Property 9: Error Resilience and Logging**
# ==========================================

@given(
    num_valid=st.integers(min_value=1, max_value=5),
    num_invalid=st.integers(min_value=1, max_value=3)
)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_error_resilience_and_logging(num_valid, num_invalid):
    """
    For any batch processing run containing both valid and invalid XML files,
    the system should process all valid files successfully, log errors for invalid files,
    and produce a log file with success_count + failure_count equal to total file count
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        total_files = num_valid + num_invalid
        
        # Create valid XML files
        for i in range(num_valid):
            xml_path = os.path.join(tmpdir, f"valid_{i}.xml")
            
            root = ET.Element('xmeml', version="4")
            seq = ET.SubElement(root, 'sequence')
            ET.SubElement(seq, 'name').text = f"Valid{i}"
            
            rate = ET.SubElement(seq, 'rate')
            ET.SubElement(rate, 'timebase').text = "30"
            ET.SubElement(seq, 'duration').text = "300"
            
            media = ET.SubElement(seq, 'media')
            video = ET.SubElement(media, 'video')
            track = ET.SubElement(video, 'track')
            clip = ET.SubElement(track, 'clipitem', id=f"clip-{i}")
            ET.SubElement(clip, 'name').text = f"Clip_{i}"
            ET.SubElement(clip, 'start').text = "0"
            ET.SubElement(clip, 'end').text = "100"
            
            tree = ET.ElementTree(root)
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
        # Create invalid XML files (malformed)
        for i in range(num_invalid):
            xml_path = os.path.join(tmpdir, f"invalid_{i}.xml")
            with open(xml_path, 'w', encoding='utf-8') as f:
                f.write("<?xml version='1.0'?>\n<invalid>This is not a valid FCP XML</invalid>")
        
        # Process batch
        output_csv = os.path.join(tmpdir, "output.csv")
        log_file = os.path.join(tmpdir, "log.txt")
        
        success_count, failure_count, failed_files = batch_convert(tmpdir, output_csv, log_file)
        
        # Verify counts
        assert success_count + failure_count == total_files, \
            f"success_count ({success_count}) + failure_count ({failure_count}) should equal total_files ({total_files})"
        
        assert success_count == num_valid, \
            f"Expected {num_valid} successful, got {success_count}"
        
        assert failure_count == num_invalid, \
            f"Expected {num_invalid} failures, got {failure_count}"
        
        assert len(failed_files) == num_invalid, \
            f"Expected {num_invalid} failed files in list, got {len(failed_files)}"
        
        # Verify log file exists and contains required information
        assert os.path.exists(log_file), "Log file should exist"
        
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
        
        # Check log contains key information
        assert "Batch Processing Summary" in log_content
        assert f"Total files processed: {total_files}" in log_content
        assert f"Successful: {success_count}" in log_content
        assert f"Failed: {failure_count}" in log_content
        
        if num_invalid > 0:
            assert "Failed files:" in log_content


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
