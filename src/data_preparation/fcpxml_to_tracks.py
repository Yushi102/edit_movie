"""
FCPXML/XMEML Parser - Extract Track Information for Multi-Track Transformer

This script parses Final Cut Pro X XML (FCPXML) and Final Cut Pro 7 XML (XMEML) files
and extracts editing track information including clip positions, scales, crops, and asset IDs.

Output format matches the expected input for the Multi-Track Transformer model:
- 20 tracks × 12 parameters = 240 dimensions per timestep
- Parameters: [active, asset_id, scale, x, y, anchor_x, anchor_y, rotation, crop_l, crop_r, crop_t, crop_b]
"""
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FCPXMLParser:
    """Parser for Final Cut Pro X XML (FCPXML) and Final Cut Pro 7 XML (XMEML) files"""
    
    def __init__(self, max_tracks: int = 20, fps: float = 10.0):
        """
        Initialize FCPXML/XMEML Parser
        
        Args:
            max_tracks: Maximum number of tracks to extract (default: 20)
            fps: Frames per second for sampling (default: 10.0)
        """
        self.max_tracks = max_tracks
        self.fps = fps
        self.frame_duration = 1.0 / fps
        
        # Asset ID mapping (clip name -> asset_id)
        self.asset_mapping = {}
        self.next_asset_id = 0
        
        # XML format detection
        self.xml_format = None  # Will be 'fcpxml' or 'xmeml'
        
        logger.info(f"FCPXMLParser initialized: max_tracks={max_tracks}, fps={fps}")
    
    def parse_time(self, time_str: str, timebase: int = None) -> float:
        """
        Parse FCPXML/XMEML time format to seconds
        
        Args:
            time_str: Time string in format "1234/2400s" (FCPXML) or frame number (XMEML)
            timebase: Timebase for XMEML format (frames per second)
        
        Returns:
            Time in seconds
        """
        if not time_str:
            return 0.0
        
        # FCPXML format: "1234/2400s"
        if isinstance(time_str, str) and time_str.endswith('s'):
            time_str = time_str.rstrip('s')
            if time_str == '0':
                return 0.0
            if '/' in time_str:
                numerator, denominator = time_str.split('/')
                return float(numerator) / float(denominator)
            else:
                return float(time_str)
        
        # XMEML format: frame number
        try:
            frames = int(time_str)
            if timebase and timebase > 0:
                return frames / float(timebase)
            else:
                return frames / 30.0  # Default to 30fps if no timebase
        except (ValueError, TypeError):
            return 0.0
    
    def get_asset_id(self, clip_name: str) -> int:
        """
        Get or create asset ID for a clip name
        
        Args:
            clip_name: Name of the clip/asset
        
        Returns:
            Asset ID (0-9, wraps around if more than 10 unique assets)
        """
        if clip_name not in self.asset_mapping:
            self.asset_mapping[clip_name] = self.next_asset_id % 10
            self.next_asset_id += 1
        
        return self.asset_mapping[clip_name]
    
    def extract_clip_info_xmeml(self, clipitem: ET.Element, track_index: int, timebase: int) -> Dict:
        """
        Extract information from an XMEML clipitem element
        
        Args:
            clipitem: XML element representing a clipitem
            track_index: Index of the track this clip belongs to
            timebase: Timebase (frames per second) from the sequence
        
        Returns:
            Dict with clip information
        """
        # Get clip name
        name_elem = clipitem.find('name')
        clip_name = name_elem.text if name_elem is not None and name_elem.text else f'clip_{track_index}'
        
        # Get timing information
        start_elem = clipitem.find('start')
        end_elem = clipitem.find('end')
        in_elem = clipitem.find('in')
        out_elem = clipitem.find('out')
        
        start_frame = int(start_elem.text) if start_elem is not None and start_elem.text else 0
        end_frame = int(end_elem.text) if end_elem is not None and end_elem.text else 0
        in_frame = int(in_elem.text) if in_elem is not None and in_elem.text else 0
        out_frame = int(out_elem.text) if out_elem is not None and out_elem.text else 0
        
        start_time = self.parse_time(str(start_frame), timebase)
        end_time = self.parse_time(str(end_frame), timebase)
        duration = end_time - start_time
        
        # Get asset ID
        asset_id = self.get_asset_id(clip_name)
        
        # Extract transform parameters (scale, position, rotation, anchor) from filters
        scale = 1.0
        pos_x = 0.0
        pos_y = 0.0
        anchor_x = 0.0
        anchor_y = 0.0
        rotation = 0.0
        
        # Look for filter effects
        for filter_elem in clipitem.findall('.//filter'):
            for effect in filter_elem.findall('.//effect'):
                effect_name = effect.find('name')
                if effect_name is not None and effect_name.text:
                    name = effect_name.text.lower()
                    
                    # Look for parameters
                    for param in effect.findall('.//parameter'):
                        param_name_elem = param.find('name')
                        param_value_elem = param.find('value')
                        param_id_elem = param.find('parameterid')
                        
                        if param_name_elem is not None:
                            param_name = param_name_elem.text.lower() if param_name_elem.text else ''
                            param_id = param_id_elem.text.lower() if param_id_elem is not None and param_id_elem.text else ''
                            
                            # Handle rotation (single value)
                            if param_id == 'rotation' or param_name == 'rotation':
                                if param_value_elem is not None and param_value_elem.text:
                                    try:
                                        rotation = float(param_value_elem.text)
                                    except (ValueError, TypeError):
                                        pass
                            
                            # Handle scale (single value)
                            elif param_id == 'scale' or param_name == 'scale':
                                if param_value_elem is not None and param_value_elem.text:
                                    try:
                                        scale = float(param_value_elem.text) / 100.0  # XMEML often uses percentage
                                    except (ValueError, TypeError):
                                        pass
                            
                            # Handle center position (horiz/vert values)
                            elif param_id == 'center' or param_name == 'center':
                                value_elem = param.find('value')
                                if value_elem is not None:
                                    horiz_elem = value_elem.find('horiz')
                                    vert_elem = value_elem.find('vert')
                                    if horiz_elem is not None and horiz_elem.text:
                                        try:
                                            pos_x = float(horiz_elem.text)
                                        except (ValueError, TypeError):
                                            pass
                                    if vert_elem is not None and vert_elem.text:
                                        try:
                                            pos_y = float(vert_elem.text)
                                        except (ValueError, TypeError):
                                            pass
                            
                            # Handle anchor point (horiz/vert values)
                            elif param_id == 'centeroffset' or 'anchor' in param_name:
                                value_elem = param.find('value')
                                if value_elem is not None:
                                    horiz_elem = value_elem.find('horiz')
                                    vert_elem = value_elem.find('vert')
                                    if horiz_elem is not None and horiz_elem.text:
                                        try:
                                            anchor_x = float(horiz_elem.text)
                                        except (ValueError, TypeError):
                                            pass
                                    if vert_elem is not None and vert_elem.text:
                                        try:
                                            anchor_y = float(vert_elem.text)
                                        except (ValueError, TypeError):
                                            pass
        
        # Extract crop parameters
        crop_l = 0.0
        crop_r = 0.0
        crop_t = 0.0
        crop_b = 0.0
        
        # Look for crop parameters in the same filter
        for filter_elem in clipitem.findall('.//filter'):
            for effect in filter_elem.findall('.//effect'):
                for param in effect.findall('.//parameter'):
                    param_id_elem = param.find('parameterid')
                    param_name_elem = param.find('name')
                    param_value_elem = param.find('value')
                    
                    if param_id_elem is not None and param_value_elem is not None:
                        param_id = param_id_elem.text.lower() if param_id_elem.text else ''
                        param_name = param_name_elem.text.lower() if param_name_elem is not None and param_name_elem.text else ''
                        
                        try:
                            param_value = float(param_value_elem.text)
                            
                            if param_id == 'leftcrop' or 'left' in param_name:
                                crop_l = param_value
                            elif param_id == 'rightcrop' or 'right' in param_name:
                                crop_r = param_value
                            elif param_id == 'topcrop' or 'top' in param_name:
                                crop_t = param_value
                            elif param_id == 'bottomcrop' or 'bottom' in param_name:
                                crop_b = param_value
                        except (ValueError, TypeError):
                            pass
        
        # Check if enabled
        enabled_elem = clipitem.find('enabled')
        enabled = enabled_elem is None or enabled_elem.text != 'FALSE'
        
        return {
            'track_index': track_index,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'asset_id': asset_id,
            'scale': scale,
            'pos_x': pos_x,
            'pos_y': pos_y,
            'anchor_x': anchor_x,
            'anchor_y': anchor_y,
            'rotation': rotation,
            'crop_l': crop_l,
            'crop_r': crop_r,
            'crop_t': crop_t,
            'crop_b': crop_b,
            'clip_name': clip_name,
            'clip_ref': '',
            'enabled': enabled,
            'source_start': self.parse_time(str(in_frame), timebase),
            'source_duration': self.parse_time(str(out_frame - in_frame), timebase),
            'graphics_text': ''
        }
    
    def extract_clip_info(self, clip_element: ET.Element, track_index: int) -> Dict:
        """
        Extract information from a single clip element
        
        Args:
            clip_element: XML element representing a clip
            track_index: Index of the track this clip belongs to
        
        Returns:
            Dict with clip information
        """
        # Get basic timing
        offset = self.parse_time(clip_element.get('offset', '0s'))
        duration = self.parse_time(clip_element.get('duration', '0s'))
        start = self.parse_time(clip_element.get('start', '0s'))
        
        # Get clip name and reference info
        clip_name = clip_element.get('name', f'clip_{track_index}')
        clip_ref = clip_element.get('ref', '')
        asset_id = self.get_asset_id(clip_name)
        
        # Extract source file info (used/unused portions)
        source_start = start
        source_duration = duration
        enabled = clip_element.get('enabled', '1') == '1'
        
        # Extract text/graphics content
        text_content = []
        for title in clip_element.findall('.//title'):
            title_name = title.get('name', '')
            if title_name:
                text_content.append(title_name)
            # Look for text elements
            for text_elem in title.findall('.//text'):
                if text_elem.text:
                    text_content.append(text_elem.text)
        
        # Also check for text in param elements
        for param in clip_element.findall('.//param'):
            if param.get('name', '').lower() in ['text', 'title', 'caption']:
                param_value = param.get('value', '')
                if param_value:
                    text_content.append(param_value)
        
        graphics_text = ' | '.join(text_content) if text_content else ''
        
        # Extract transform parameters (scale, position, rotation, anchor)
        scale = 1.0
        pos_x = 0.0
        pos_y = 0.0
        anchor_x = 0.0
        anchor_y = 0.0
        rotation = 0.0
        
        # Look for transform parameters in video elements
        for video in clip_element.findall('.//video'):
            # Check for param elements
            for param in video.findall('.//param'):
                param_name = param.get('name', '')
                param_value = param.get('value', '0')
                
                try:
                    value = float(param_value)
                    if 'scale' in param_name.lower():
                        scale = value
                    elif 'rotation' in param_name.lower():
                        rotation = value
                    elif 'anchor' in param_name.lower():
                        if 'x' in param_name.lower():
                            anchor_x = value
                        elif 'y' in param_name.lower():
                            anchor_y = value
                    elif 'x' in param_name.lower() and 'position' in param_name.lower():
                        pos_x = value
                    elif 'y' in param_name.lower() and 'position' in param_name.lower():
                        pos_y = value
                except ValueError:
                    pass
        
        # Extract crop parameters
        crop_l = 0.0
        crop_r = 0.0
        crop_t = 0.0
        crop_b = 0.0
        
        # Look for crop/trim parameters
        for param in clip_element.findall('.//param'):
            param_name = param.get('name', '').lower()
            param_value = param.get('value', '0')
            
            try:
                value = float(param_value)
                if 'crop' in param_name or 'trim' in param_name:
                    if 'left' in param_name:
                        crop_l = value
                    elif 'right' in param_name:
                        crop_r = value
                    elif 'top' in param_name:
                        crop_t = value
                    elif 'bottom' in param_name:
                        crop_b = value
            except ValueError:
                pass
        
        return {
            'track_index': track_index,
            'start_time': offset,
            'end_time': offset + duration,
            'duration': duration,
            'asset_id': asset_id,
            'scale': scale,
            'pos_x': pos_x,
            'pos_y': pos_y,
            'anchor_x': anchor_x,
            'anchor_y': anchor_y,
            'rotation': rotation,
            'crop_l': crop_l,
            'crop_r': crop_r,
            'crop_t': crop_t,
            'crop_b': crop_b,
            'clip_name': clip_name,
            'clip_ref': clip_ref,
            'enabled': enabled,
            'source_start': source_start,
            'source_duration': source_duration,
            'graphics_text': graphics_text
        }
    
    def parse_xmeml(self, xml_path: str) -> Tuple[List[Dict], float]:
        """
        Parse XMEML file and extract all clips
        
        Args:
            xml_path: Path to XMEML file
        
        Returns:
            Tuple of (list of clip dicts, total duration in seconds)
        """
        logger.info(f"Parsing XMEML: {xml_path}")
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        clips = []
        max_end_time = 0.0
        
        # Find all sequences
        for sequence in root.findall('.//sequence'):
            # Get timebase from rate
            timebase = 30  # Default
            rate_elem = sequence.find('.//rate/timebase')
            if rate_elem is not None and rate_elem.text:
                try:
                    timebase = int(rate_elem.text)
                except ValueError:
                    pass
            
            logger.info(f"  Sequence timebase: {timebase} fps")
            
            # Get sequence duration
            duration_elem = sequence.find('duration')
            if duration_elem is not None and duration_elem.text:
                try:
                    duration_frames = int(duration_elem.text)
                    max_end_time = self.parse_time(str(duration_frames), timebase)
                except ValueError:
                    pass
            
            # Find all video tracks
            media = sequence.find('media')
            if media is not None:
                video = media.find('video')
                if video is not None:
                    tracks = video.findall('track')
                    
                    for track_index, track in enumerate(tracks):
                        if track_index >= self.max_tracks:
                            logger.warning(f"Reached max tracks ({self.max_tracks}), skipping remaining tracks")
                            break
                        
                        # Process all clipitems in this track
                        for clipitem in track.findall('.//clipitem'):
                            clip_info = self.extract_clip_info_xmeml(clipitem, track_index, timebase)
                            clips.append(clip_info)
                            
                            max_end_time = max(max_end_time, clip_info['end_time'])
        
        logger.info(f"Extracted {len(clips)} clips, total duration: {max_end_time:.2f}s")
        logger.info(f"Unique assets: {len(self.asset_mapping)}")
        
        return clips, max_end_time
    
    def parse_fcpxml(self, xml_path: str) -> Tuple[List[Dict], float]:
        """
        Parse FCPXML file and extract all clips
        
        Args:
            xml_path: Path to FCPXML file
        
        Returns:
            Tuple of (list of clip dicts, total duration in seconds)
        """
        logger.info(f"Parsing FCPXML: {xml_path}")
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        clips = []
        max_end_time = 0.0
        
        # Find all sequences/projects
        for sequence in root.findall('.//sequence'):
            # Find all spine elements (main timeline)
            for spine in sequence.findall('.//spine'):
                track_index = 0
                
                # Process all clips in the spine
                for clip in spine.findall('.//clip'):
                    if track_index >= self.max_tracks:
                        logger.warning(f"Reached max tracks ({self.max_tracks}), skipping remaining clips")
                        break
                    
                    clip_info = self.extract_clip_info(clip, track_index)
                    clips.append(clip_info)
                    
                    max_end_time = max(max_end_time, clip_info['end_time'])
                    track_index += 1
                
                # Also check for clips in video/audio lanes
                for lane in spine.findall('.//lane'):
                    for clip in lane.findall('.//clip'):
                        if track_index >= self.max_tracks:
                            break
                        
                        clip_info = self.extract_clip_info(clip, track_index)
                        clips.append(clip_info)
                        
                        max_end_time = max(max_end_time, clip_info['end_time'])
                        track_index += 1
        
        logger.info(f"Extracted {len(clips)} clips, total duration: {max_end_time:.2f}s")
        logger.info(f"Unique assets: {len(self.asset_mapping)}")
        
        return clips, max_end_time
    
    def parse_xml(self, xml_path: str) -> Tuple[List[Dict], float]:
        """
        Auto-detect XML format and parse accordingly
        
        Args:
            xml_path: Path to XML file
        
        Returns:
            Tuple of (list of clip dicts, total duration in seconds)
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Detect format
        if root.tag == 'xmeml':
            self.xml_format = 'xmeml'
            logger.info("Detected XMEML format (Final Cut Pro 7)")
            return self.parse_xmeml(xml_path)
        elif root.tag == 'fcpxml':
            self.xml_format = 'fcpxml'
            logger.info("Detected FCPXML format (Final Cut Pro X)")
            return self.parse_fcpxml(xml_path)
        else:
            logger.warning(f"Unknown XML format: {root.tag}, attempting FCPXML parser")
            self.xml_format = 'fcpxml'
            return self.parse_fcpxml(xml_path)
    
    def clips_to_track_sequence(self, clips: List[Dict], total_duration: float) -> np.ndarray:
        """
        Convert clip list to track sequence array
        
        Args:
            clips: List of clip dictionaries
            total_duration: Total duration of the sequence in seconds
        
        Returns:
            Array of shape (num_timesteps, num_tracks, 12)
            12 parameters: [active, asset_id, scale, x, y, anchor_x, anchor_y, rotation, crop_l, crop_r, crop_t, crop_b]
        """
        # Calculate number of timesteps
        num_timesteps = int(np.ceil(total_duration * self.fps)) + 1
        
        # Initialize array with zeros
        sequence = np.zeros((num_timesteps, self.max_tracks, 12), dtype=np.float32)
        
        # Fill in clip data
        for clip in clips:
            track_idx = clip['track_index']
            if track_idx >= self.max_tracks:
                continue
            
            # Calculate timestep range for this clip
            start_timestep = int(clip['start_time'] * self.fps)
            end_timestep = int(clip['end_time'] * self.fps)
            
            # Ensure within bounds
            start_timestep = max(0, start_timestep)
            end_timestep = min(num_timesteps, end_timestep)
            
            # Fill in parameters for all timesteps in this clip
            for t in range(start_timestep, end_timestep):
                sequence[t, track_idx, 0] = 1.0  # active
                sequence[t, track_idx, 1] = float(clip['asset_id'])
                sequence[t, track_idx, 2] = clip['scale']
                sequence[t, track_idx, 3] = clip['pos_x']
                sequence[t, track_idx, 4] = clip['pos_y']
                sequence[t, track_idx, 5] = clip['anchor_x']
                sequence[t, track_idx, 6] = clip['anchor_y']
                sequence[t, track_idx, 7] = clip['rotation']
                sequence[t, track_idx, 8] = clip['crop_l']
                sequence[t, track_idx, 9] = clip['crop_r']
                sequence[t, track_idx, 10] = clip['crop_t']
                sequence[t, track_idx, 11] = clip['crop_b']
        
        logger.info(f"Created sequence: shape={sequence.shape}")
        return sequence
    
    def save_sequence(self, sequence: np.ndarray, output_path: str, video_id: str):
        """
        Save sequence to NPZ file
        
        Args:
            sequence: Track sequence array
            output_path: Path to save NPZ file
            video_id: Video identifier
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Flatten to (num_timesteps, 240) for model input (20 tracks × 12 parameters)
        num_timesteps = sequence.shape[0]
        flattened = sequence.reshape(num_timesteps, -1)
        
        # Save with metadata
        np.savez_compressed(
            output_path,
            sequences=flattened,
            video_ids=[video_id],
            num_tracks=self.max_tracks,
            fps=self.fps,
            asset_mapping=self.asset_mapping
        )
        
        logger.info(f"Saved sequence to {output_path}")
        logger.info(f"  Shape: {flattened.shape}")
        logger.info(f"  Video ID: {video_id}")
    
    def save_csv(self, sequence: np.ndarray, output_path: str, video_id: str, clips: List[Dict] = None):
        """
        Save sequence to CSV file for inspection
        
        Args:
            sequence: Track sequence array
            output_path: Path to save CSV file
            video_id: Video identifier
            clips: Optional list of clip dicts for detailed info
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame
        num_timesteps = sequence.shape[0]
        rows = []
        
        for t in range(num_timesteps):
            time = t * self.frame_duration
            for track_idx in range(self.max_tracks):
                params = sequence[t, track_idx]
                
                # Find matching clip for this timestep and track
                clip_name = ''
                clip_ref = ''
                enabled = True
                source_start = 0.0
                source_duration = 0.0
                graphics_text = ''
                
                if clips:
                    for clip in clips:
                        if (clip['track_index'] == track_idx and 
                            clip['start_time'] <= time < clip['end_time']):
                            clip_name = clip['clip_name']
                            clip_ref = clip.get('clip_ref', '')
                            enabled = clip.get('enabled', True)
                            source_start = clip.get('source_start', 0.0)
                            source_duration = clip.get('source_duration', 0.0)
                            graphics_text = clip.get('graphics_text', '')
                            break
                
                rows.append({
                    'video_id': video_id,
                    'time': time,
                    'track': track_idx,
                    'active': int(params[0]),
                    'asset_id': int(params[1]),
                    'clip_name': clip_name,
                    'clip_ref': clip_ref,
                    'enabled': enabled,
                    'source_start': source_start,
                    'source_duration': source_duration,
                    'graphics_text': graphics_text,
                    'scale': params[2],
                    'pos_x': params[3],
                    'pos_y': params[4],
                    'anchor_x': params[5],
                    'anchor_y': params[6],
                    'rotation': params[7],
                    'crop_l': params[8],
                    'crop_r': params[9],
                    'crop_t': params[10],
                    'crop_b': params[11]
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved CSV to {output_path}")
        logger.info(f"  Rows: {len(df)}")


def main():
    parser = argparse.ArgumentParser(description='Extract track information from FCPXML files')
    parser.add_argument('input', type=str, help='Path to FCPXML file')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--video-id', type=str, default=None, help='Video ID (default: filename)')
    parser.add_argument('--max-tracks', type=int, default=20, help='Maximum number of tracks')
    parser.add_argument('--fps', type=float, default=10.0, help='Sampling rate (frames per second)')
    parser.add_argument('--format', type=str, choices=['npz', 'csv', 'both'], default='both',
                        help='Output format')
    
    args = parser.parse_args()
    
    # Determine video ID
    video_id = args.video_id
    if video_id is None:
        video_id = Path(args.input).stem
    
    # Create parser
    fcpxml_parser = FCPXMLParser(max_tracks=args.max_tracks, fps=args.fps)
    
    # Parse XML (auto-detect format)
    clips, total_duration = fcpxml_parser.parse_xml(args.input)
    
    # Convert to sequence
    sequence = fcpxml_parser.clips_to_track_sequence(clips, total_duration)
    
    # Save outputs
    output_dir = Path(args.output)
    
    if args.format in ['npz', 'both']:
        npz_path = output_dir / f'{video_id}_tracks.npz'
        fcpxml_parser.save_sequence(sequence, str(npz_path), video_id)
    
    if args.format in ['csv', 'both']:
        csv_path = output_dir / f'{video_id}_tracks.csv'
        fcpxml_parser.save_csv(sequence, str(csv_path), video_id, clips)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("Extraction Summary")
    logger.info("="*70)
    logger.info(f"Input: {args.input}")
    logger.info(f"Video ID: {video_id}")
    logger.info(f"Duration: {total_duration:.2f}s")
    logger.info(f"Timesteps: {sequence.shape[0]}")
    logger.info(f"Tracks: {args.max_tracks}")
    logger.info(f"FPS: {args.fps}")
    logger.info(f"Total clips: {len(clips)}")
    logger.info(f"Unique assets: {len(fcpxml_parser.asset_mapping)}")
    logger.info("\nAsset Mapping:")
    for asset_name, asset_id in sorted(fcpxml_parser.asset_mapping.items(), key=lambda x: x[1]):
        logger.info(f"  {asset_id}: {asset_name}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
