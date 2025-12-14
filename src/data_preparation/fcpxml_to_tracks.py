"""
FCPXML Parser - Extract Track Information for Multi-Track Transformer

This script parses Final Cut Pro X XML files and extracts editing track information
including clip positions, scales, crops, and asset IDs.

Output format matches the expected input for the Multi-Track Transformer model:
- 20 tracks × 9 parameters = 180 dimensions per timestep
- Parameters: [active, asset_id, scale, x, y, crop_l, crop_r, crop_t, crop_b]
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
    """Parser for Final Cut Pro X XML files"""
    
    def __init__(self, max_tracks: int = 20, fps: float = 10.0):
        """
        Initialize FCPXML Parser
        
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
        
        logger.info(f"FCPXMLParser initialized: max_tracks={max_tracks}, fps={fps}")
    
    def parse_time(self, time_str: str) -> float:
        """
        Parse FCPXML time format to seconds
        
        Args:
            time_str: Time string in format "1234/2400s" (frames/framerate)
        
        Returns:
            Time in seconds
        """
        if not time_str or time_str == '0s':
            return 0.0
        
        # Remove 's' suffix
        time_str = time_str.rstrip('s')
        
        # Parse fraction
        if '/' in time_str:
            numerator, denominator = time_str.split('/')
            return float(numerator) / float(denominator)
        else:
            return float(time_str)
    
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
        
        # Extract transform parameters (scale, position)
        scale = 1.0
        pos_x = 0.0
        pos_y = 0.0
        
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
    
    def clips_to_track_sequence(self, clips: List[Dict], total_duration: float) -> np.ndarray:
        """
        Convert clip list to track sequence array
        
        Args:
            clips: List of clip dictionaries
            total_duration: Total duration of the sequence in seconds
        
        Returns:
            Array of shape (num_timesteps, num_tracks, 9)
            9 parameters: [active, asset_id, scale, x, y, crop_l, crop_r, crop_t, crop_b]
        """
        # Calculate number of timesteps
        num_timesteps = int(np.ceil(total_duration * self.fps)) + 1
        
        # Initialize array with zeros
        sequence = np.zeros((num_timesteps, self.max_tracks, 9), dtype=np.float32)
        
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
                sequence[t, track_idx, 5] = clip['crop_l']
                sequence[t, track_idx, 6] = clip['crop_r']
                sequence[t, track_idx, 7] = clip['crop_t']
                sequence[t, track_idx, 8] = clip['crop_b']
        
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
        
        # Flatten to (num_timesteps, 180) for model input
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
                    'crop_l': params[5],
                    'crop_r': params[6],
                    'crop_t': params[7],
                    'crop_b': params[8]
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
    
    # Parse FCPXML
    clips, total_duration = fcpxml_parser.parse_fcpxml(args.input)
    
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
