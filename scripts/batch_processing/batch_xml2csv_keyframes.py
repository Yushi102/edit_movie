"""
XML to CSV converter for Premiere Pro timeline data
"""
import pandas as pd
import xml.etree.ElementTree as ET
import argparse
import os
import glob
import re
import logging
from typing import Tuple, Dict, List, Any, Optional
import urllib.parse

# Configuration
INTERVAL = 0.1  # Sample every 0.1 seconds

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def clean_path(raw_url: str) -> str:
    """Extract clean file path from XML pathurl element"""
    if not raw_url:
        return ""
    decoded = urllib.parse.unquote(raw_url)
    decoded = decoded.replace("file://localhost/", "").replace("file://", "")
    if os.name == 'nt' and decoded.startswith("/") and ':' in decoded:
        decoded = decoded.lstrip("/")
    return decoded.replace("/", "\\") if os.name == 'nt' else decoded


def extract_source_video_name(xml_path: str) -> str:
    """Extract source video filename from XML"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for pathurl in root.iter("pathurl"):
            if pathurl.text:
                clean_url = clean_path(pathurl.text)
                if clean_url.lower().endswith(('.mp4', '.mov', '.mkv', '.avi', '.m4v')):
                    basename = os.path.basename(clean_url)
                    return os.path.splitext(basename)[0]
        logger.warning(f"No video file found in XML: {xml_path}")
        return "unknown"
    except Exception as e:
        logger.error(f"Error extracting source video name: {e}")
        return "unknown"


def classify_asset_id(clip_name: str, track_num: int) -> int:
    """Classify AssetID based on clip name and track number"""
    if track_num == 1:
        return 0  # Game footage
    if track_num == 2:
        return 1  # Face camera
    if track_num >= 3:
        # Check for image extensions
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp']
        clip_name_lower = clip_name.lower()
        for ext in image_extensions:
            if ext in clip_name_lower:
                return 3  # Image asset
        # Check for text keywords
        text_keywords = ['text', 'title', 'caption', 'subtitle', 'テキスト', 'タイトル', '字幕']
        for keyword in text_keywords:
            if keyword.lower() in clip_name_lower:
                return 2  # Text graphic
        # Extract ID from clip name
        match = re.search(r'(Class|ID)[-_]?(\d+)', clip_name, re.IGNORECASE)
        if match:
            return max(2, int(match.group(2)))
        return 2  # Default to text graphic
    return 0


def parse_keyframes(parameter_node: ET.Element) -> Tuple[Any, bool]:
    """Parse keyframe information from parameter node"""
    # Check for fixed value
    val_node = parameter_node.find("value")
    if val_node is not None and val_node.text is not None:
        try:
            return float(val_node.text), False
        except ValueError:
            pass
    # Check for keyframes
    kfs = parameter_node.findall("keyframe")
    if not kfs:
        return None, False
    kf_data = []
    for kf in kfs:
        when_node = kf.find("when")
        val_node = kf.find("value")
        if when_node is not None and val_node is not None:
            try:
                time = int(when_node.text)
                val = float(val_node.text)
                kf_data.append((time, val))
            except (ValueError, TypeError):
                continue
    if not kf_data:
        return None, False
    kf_data.sort(key=lambda x: x[0])
    return kf_data, True


def interpolate_value(frame: int, kf_data: Any, default_val: float) -> float:
    """Interpolate value at given frame using linear interpolation"""
    if not isinstance(kf_data, list):
        return kf_data if kf_data is not None else default_val
    if not kf_data:
        return default_val
    if frame <= kf_data[0][0]:
        return kf_data[0][1]
    if frame >= kf_data[-1][0]:
        return kf_data[-1][1]
    for i in range(len(kf_data) - 1):
        t1, v1 = kf_data[i]
        t2, v2 = kf_data[i + 1]
        if t1 <= frame <= t2:
            if t2 == t1:
                return v1
            ratio = (frame - t1) / (t2 - t1)
            return v1 + (v2 - v1) * ratio
    return default_val


def parse_single_xml(xml_path: str) -> Optional[pd.DataFrame]:
    """Parse single XML file and generate DataFrame"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Failed to parse XML {xml_path}: {e}")
        return None
    
    sequence = root.find(".//sequence")
    if not sequence:
        logger.warning(f"No sequence found in {xml_path}")
        return None
    
    # Get FPS
    timebase = sequence.find(".//rate/timebase")
    fps = float(timebase.text) if timebase is not None else 60.0
    
    # Get total frames
    duration_node = sequence.find("duration")
    total_frames = int(duration_node.text) if duration_node is not None else 0
    
    if total_frames == 0:
        for end_tag in sequence.findall(".//clipitem/end"):
            try:
                val = int(end_tag.text)
                if val > total_frames:
                    total_frames = val
            except (ValueError, TypeError):
                continue
    
    if total_frames == 0:
        logger.warning(f"Could not determine duration for {xml_path}")
        return None
    
    source_video_name = extract_source_video_name(xml_path)
    video_node = sequence.find(".//media/video")
    if not video_node:
        logger.warning(f"No video tracks found in {xml_path}")
        return None
    
    tracks = video_node.findall("track")
    timeline_data = {}
    
    # Process each track (continued in next chunk)

    
    for t_idx, track in enumerate(tracks):
        v_num = t_idx + 1
        if v_num > 20:
            break
        prefix = f"target_v{v_num}"
        
        for clip in track.findall("clipitem"):
            start_node = clip.find("start")
            end_node = clip.find("end")
            if start_node is None or end_node is None:
                continue
            try:
                start = int(start_node.text)
                end = int(end_node.text)
            except (ValueError, TypeError):
                continue
            if start < 0:
                start = 0
            
            name_node = clip.find("name")
            clip_name = name_node.text if name_node is not None else ""
            asset_id = classify_asset_id(clip_name, v_num)
            
            # Initialize parameters
            p_data = {
                'scale': (100.0, False), 'pos_x': (0.0, False), 'pos_y': (0.0, False),
                'crop_l': (0.0, False), 'crop_r': (0.0, False),
                'crop_t': (0.0, False), 'crop_b': (0.0, False)
            }
            
            # Parse effects (continued in next chunk)

            
            for filt in clip.findall(".//filter"):
                eff = filt.find("effect")
                if eff is None:
                    continue
                eff_name_node = eff.find("name")
                if eff_name_node is None:
                    continue
                eff_name = eff_name_node.text
                
                if eff_name == "Basic Motion":
                    for p in eff.findall(".//parameter"):
                        pid_node = p.find("parameterid")
                        if pid_node is None:
                            continue
                        pid = pid_node.text
                        
                        if pid == "scale":
                            data, is_anim = parse_keyframes(p)
                            p_data['scale'] = (data, is_anim)
                        elif pid == "center":
                            val_node = p.find("value")
                            if val_node:
                                h = val_node.find("horiz")
                                v = val_node.find("vert")
                                if h is not None:
                                    try:
                                        p_data['pos_x'] = (float(h.text), False)
                                    except (ValueError, TypeError):
                                        pass
                                if v is not None:
                                    try:
                                        p_data['pos_y'] = (float(v.text), False)
                                    except (ValueError, TypeError):
                                        pass
                            # Handle keyframes for center
                            kfs = p.findall("keyframe")
                            if kfs:
                                kf_x, kf_y = [], []
                                for kf in kfs:
                                    when_node = kf.find("when")
                                    val = kf.find("value")
                                    if when_node is not None and val:
                                        try:
                                            t = int(when_node.text)
                                            h = val.find("horiz")
                                            v = val.find("vert")
                                            if h is not None:
                                                kf_x.append((t, float(h.text)))
                                            if v is not None:
                                                kf_y.append((t, float(v.text)))
                                        except (ValueError, TypeError):
                                            continue
                                kf_x.sort(key=lambda x: x[0])
                                kf_y.sort(key=lambda x: x[0])
                                if kf_x:
                                    p_data['pos_x'] = (kf_x, True)
                                if kf_y:
                                    p_data['pos_y'] = (kf_y, True)
                
                elif eff_name == "Crop":
                    for p in eff.findall(".//parameter"):
                        pid_node = p.find("parameterid")
                        if pid_node is None:
                            continue
                        pid = pid_node.text
                        data, is_anim = parse_keyframes(p)
                        if pid == "left":
                            p_data['crop_l'] = (data, is_anim)
                        elif pid == "right":
                            p_data['crop_r'] = (data, is_anim)
                        elif pid == "top":
                            p_data['crop_t'] = (data, is_anim)
                        elif pid == "bottom":
                            p_data['crop_b'] = (data, is_anim)
            
            # Record frame data (continued in next chunk)

            
            for f in range(start, end):
                if f not in timeline_data:
                    timeline_data[f] = {}
                rel_frame = f - start
                timeline_data[f][f"{prefix}_active"] = 1
                timeline_data[f][f"{prefix}_asset"] = asset_id
                timeline_data[f][f"{prefix}_scale"] = interpolate_value(
                    rel_frame, p_data['scale'][0] if p_data['scale'][1] else p_data['scale'][0], 100.0)
                timeline_data[f][f"{prefix}_x"] = interpolate_value(
                    rel_frame, p_data['pos_x'][0] if p_data['pos_x'][1] else p_data['pos_x'][0], 0.0)
                timeline_data[f][f"{prefix}_y"] = interpolate_value(
                    rel_frame, p_data['pos_y'][0] if p_data['pos_y'][1] else p_data['pos_y'][0], 0.0)
                timeline_data[f][f"{prefix}_crop_l"] = interpolate_value(
                    rel_frame, p_data['crop_l'][0] if p_data['crop_l'][1] else p_data['crop_l'][0], 0.0)
                timeline_data[f][f"{prefix}_crop_r"] = interpolate_value(
                    rel_frame, p_data['crop_r'][0] if p_data['crop_r'][1] else p_data['crop_r'][0], 0.0)
                timeline_data[f][f"{prefix}_crop_t"] = interpolate_value(
                    rel_frame, p_data['crop_t'][0] if p_data['crop_t'][1] else p_data['crop_t'][0], 0.0)
                timeline_data[f][f"{prefix}_crop_b"] = interpolate_value(
                    rel_frame, p_data['crop_b'][0] if p_data['crop_b'][1] else p_data['crop_b'][0], 0.0)
    
    # Generate CSV rows
    rows = []
    max_sec = total_frames / fps
    num_samples = int(max_sec / INTERVAL)
    video_id = os.path.splitext(os.path.basename(xml_path))[0]
    
    for i in range(num_samples):
        sec = i * INTERVAL
        frame = int(sec * fps)
        row = {'video_id': video_id, 'source_video_name': source_video_name, 'time': round(sec, 2)}
        for t in range(1, 21):
            prefix = f"target_v{t}"
            data = timeline_data.get(frame, {})
            row[f"{prefix}_active"] = data.get(f"{prefix}_active", 0)
            row[f"{prefix}_asset"] = data.get(f"{prefix}_asset", 0)
            row[f"{prefix}_scale"] = data.get(f"{prefix}_scale", 100.0)
            row[f"{prefix}_x"] = data.get(f"{prefix}_x", 0.0)
            row[f"{prefix}_y"] = data.get(f"{prefix}_y", 0.0)
            row[f"{prefix}_crop_l"] = data.get(f"{prefix}_crop_l", 0.0)
            row[f"{prefix}_crop_r"] = data.get(f"{prefix}_crop_r", 0.0)
            row[f"{prefix}_crop_t"] = data.get(f"{prefix}_crop_t", 0.0)
            row[f"{prefix}_crop_b"] = data.get(f"{prefix}_crop_b", 0.0)
        rows.append(row)
    
    logger.info(f"Successfully parsed {xml_path}: {len(rows)} rows generated")
    return pd.DataFrame(rows)



def batch_convert(xml_folder: str, output_csv: str, log_file: str = "batch_processing.log") -> Tuple[int, int, List[str]]:
    """Batch convert all XML files in folder"""
    xml_files = glob.glob(os.path.join(xml_folder, "*.xml"))
    logger.info(f"Found {len(xml_files)} XML files in {xml_folder}")
    
    all_dfs = []
    failed_files = []
    success_count = 0
    failure_count = 0
    
    for xml_file in xml_files:
        logger.info(f"Processing: {os.path.basename(xml_file)}...")
        df = parse_single_xml(xml_file)
        if df is not None and not df.empty:
            all_dfs.append(df)
            success_count += 1
        else:
            failed_files.append(os.path.basename(xml_file))
            failure_count += 1
            logger.error(f"Failed to process: {xml_file}")
    
    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        master_df.to_csv(output_csv, index=False)
        logger.info(f"Master dataset saved to: {output_csv}")
        logger.info(f"Total rows: {len(master_df)}")
    else:
        logger.error("No data extracted from any XML files")
    
    # Write log file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Batch Processing Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total files processed: {len(xml_files)}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {failure_count}\n")
        f.write(f"Success rate: {success_count / len(xml_files) * 100:.2f}%\n" if xml_files else "N/A\n")
        f.write("\n")
        if failed_files:
            f.write("Failed files:\n")
            for fname in failed_files:
                f.write(f"  - {fname}\n")
        else:
            f.write("All files processed successfully!\n")
        f.write("\n")
        f.write(f"Output CSV: {output_csv}\n")
        if all_dfs:
            f.write(f"Total rows generated: {len(master_df)}\n")
    
    logger.info(f"Processing log saved to: {log_file}")
    logger.info(f"✅ Batch processing complete: {success_count} succeeded, {failure_count} failed")
    return success_count, failure_count, failed_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert Premiere Pro XML files to CSV training data")
    parser.add_argument("xml_folder", help="Folder containing XML files")
    parser.add_argument("--output", default="master_training_data.csv", help="Output CSV filename")
    parser.add_argument("--log", default="batch_processing.log", help="Log file path")
    args = parser.parse_args()
    batch_convert(args.xml_folder, args.output, args.log)
