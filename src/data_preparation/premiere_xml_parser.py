"""
Premiere Pro XML Parser - Extract Track Information for Multi-Track Transformer

This script parses Premiere Pro XML (XMEML) files and extracts editing track information
including clip positions, scales, crops, and asset IDs.

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


class PremiereXMLParser:
    """Parser for Premiere Pro XML (XMEML) files"""
    
    def __init__(self, max_tracks: int = 20, fps: float = 10.0):
        """
        Initialize Premiere XML Parser
        
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
        
        # Store sequence timebase for time conversion
        self.timebase = 30.0
        
        logger.info(f"PremiereXMLParser initialized: max_tracks={max_tracks}, fps={fps}")
    
    def parse_time(self, frames: str, timebase: float = None) -> float:
        """
        Convert frame count to seconds
        
        Args:
            frames: Frame count as string
            timebase: Frames per second (uses sequence timebase if not provided)
        
        Returns:
            Time in seconds
        """
        if not frames:
            return 0.0
        
        try:
            frame_count = int(frames)
            tb = timebase if timebase else self.timebase
            return frame_count / tb
        except (ValueError, ZeroDivisionError):
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
    
    def extract_clip_info(self, clipitem: ET.Element, track_index: int) -> Dict:
        """
        Extract information from a single clipitem element
        
        Args:
            clipitem: XML element representing a clip item
            track_index: Index of the track this clip belongs to
        
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
        
        start_time = self.parse_time(start_elem.text if start_elem is not None else '0')
        end_time = self.parse_time(end_elem.text if end_elem is not None else '0')
        source_start = self.parse_time(in_elem.text if in_elem is not None else '0')
        source_end = self.parse_time(out_elem.text if out_elem is not None else '0')
        
        duration = end_time - start_time
        source_duration = source_end - source_start
        
        # Get asset ID
        asset_id = self.get_asset_id(clip_name)
        
        # Check if clip is enabled
        enabled_elem = clipitem.find('enabled')
        enabled = enabled_elem is None or enabled_elem.text != 'FALSE'
        
        # Extract transform parameters (scale, position, anchor, rotation)
        scale = 1.0
        pos_x = 0.0
        pos_y = 0.0
        anchor_x = 0.0
        anchor_y = 0.0
        rotation = 0.0
        
        # Track keyframe information
        has_keyframes = False
        keyframe_times = []
        
        # Look for filter effects (motion, transform)
        for effect in clipitem.findall('.//effect'):
            effect_name_elem = effect.find('name')
            if effect_name_elem is not None:
                effect_name = effect_name_elem.text.lower() if effect_name_elem.text else ''
                
                if 'motion' in effect_name or 'transform' in effect_name:
                    # Look for parameters
                    for param in effect.findall('.//parameter'):
                        param_id_elem = param.find('parameterid')
                        
                        if param_id_elem is not None:
                            param_id = param_id_elem.text.lower() if param_id_elem.text else ''
                            
                            # Check for keyframes
                            keyframe_elems = param.findall('.//keyframe')
                            if keyframe_elems:
                                has_keyframes = True
                                # Get the first keyframe value (or could interpolate)
                                first_keyframe = keyframe_elems[0]
                                value_elem = first_keyframe.find('.//value')
                                when_elem = first_keyframe.find('when')
                                
                                if when_elem is not None and when_elem.text:
                                    keyframe_time = self.parse_time(when_elem.text)
                                    keyframe_times.append(keyframe_time)
                            else:
                                # No keyframes, use static value
                                value_elem = param.find('.//value')
                            
                            if value_elem is not None:
                                try:
                                    value = float(value_elem.text)
                                    if 'scale' in param_id:
                                        scale = value / 100.0  # Premiere uses percentage
                                    elif 'rotation' in param_id or 'angle' in param_id:
                                        rotation = value  # Degrees
                                    elif 'anchor' in param_id:
                                        if 'horizontal' in param_id or 'x' in param_id:
                                            anchor_x = value
                                        elif 'vertical' in param_id or 'y' in param_id:
                                            anchor_y = value
                                    elif 'position' in param_id:
                                        if 'horizontal' in param_id or 'x' in param_id:
                                            pos_x = value
                                        elif 'vertical' in param_id or 'y' in param_id:
                                            pos_y = value
                                except (ValueError, TypeError):
                                    pass
        
        # Extract crop parameters
        crop_l = 0.0
        crop_r = 0.0
        crop_t = 0.0
        crop_b = 0.0
        
        for effect in clipitem.findall('.//effect'):
            effect_name_elem = effect.find('name')
            if effect_name_elem is not None:
                effect_name = effect_name_elem.text.lower() if effect_name_elem.text else ''
                
                if 'crop' in effect_name:
                    for param in effect.findall('.//parameter'):
                        param_id_elem = param.find('parameterid')
                        value_elem = param.find('.//value')
                        
                        if param_id_elem is not None and value_elem is not None:
                            param_id = param_id_elem.text.lower() if param_id_elem.text else ''
                            
                            try:
                                value = float(value_elem.text)
                                if 'left' in param_id:
                                    crop_l = value
                                elif 'right' in param_id:
                                    crop_r = value
                                elif 'top' in param_id:
                                    crop_t = value
                                elif 'bottom' in param_id:
                                    crop_b = value
                            except (ValueError, TypeError):
                                pass
        
        # Extract text/graphics content (from title clips)
        text_content = []
        for text_elem in clipitem.findall('.//text'):
            if text_elem.text:
                text_content.append(text_elem.text)
        
        # Also check for title/caption in name
        if 'title' in clip_name.lower() or 'caption' in clip_name.lower() or 'text' in clip_name.lower():
            text_content.append(clip_name)
        
        graphics_text = ' | '.join(text_content) if text_content else ''
        
        # Get clip reference ID
        clip_id = clipitem.get('id', '')
        
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
            'clip_ref': clip_id,
            'enabled': enabled,
            'source_start': source_start,
            'source_duration': source_duration,
            'graphics_text': graphics_text,
            'has_keyframes': has_keyframes,
            'keyframe_count': len(keyframe_times)
        }
    
    def parse_premiere_xml(self, xml_path: str) -> Tuple[List[Dict], float]:
        """
        Parse Premiere Pro XML file and extract all clips
        
        Args:
            xml_path: Path to Premiere XML file
        
        Returns:
            Tuple of (list of clip dicts, total duration in seconds)
        """
        logger.info(f"Parsing Premiere XML: {xml_path}")
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        clips = []
        max_end_time = 0.0
        
        # Find all sequences
        for sequence in root.findall('.//sequence'):
            # Get sequence timebase
            rate_elem = sequence.find('.//rate/timebase')
            if rate_elem is not None and rate_elem.text:
                try:
                    self.timebase = float(rate_elem.text)
                    logger.info(f"Sequence timebase: {self.timebase} fps")
                except ValueError:
                    pass
            
            # Get sequence name
            seq_name_elem = sequence.find('name')
            seq_name = seq_name_elem.text if seq_name_elem is not None and seq_name_elem.text else 'Unknown'
            logger.info(f"Processing sequence: {seq_name}")
            
            # Find all video tracks
            video_tracks = sequence.findall('.//video/track')
            
            for track_idx, track in enumerate(video_tracks):
                if track_idx >= self.max_tracks:
                    logger.warning(f"Reached max tracks ({self.max_tracks}), skipping remaining tracks")
                    break
                
                # Process all clip items in this track
                for clipitem in track.findall('.//clipitem'):
                    clip_info = self.extract_clip_info(clipitem, track_idx)
                    clips.append(clip_info)
                    
                    max_end_time = max(max_end_time, clip_info['end_time'])
        
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
            
            # Skip disabled clips
            if not clip['enabled']:
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
        
        # Flatten to (num_timesteps, 240) for model input
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
                has_keyframes = False
                keyframe_count = 0
                
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
                            has_keyframes = clip.get('has_keyframes', False)
                            keyframe_count = clip.get('keyframe_count', 0)
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
                    'has_keyframes': has_keyframes,
                    'keyframe_count': keyframe_count,
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
    parser = argparse.ArgumentParser(description='Extract track information from Premiere Pro XML files')
    parser.add_argument('input', type=str, help='Path to Premiere XML file')
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
    premiere_parser = PremiereXMLParser(max_tracks=args.max_tracks, fps=args.fps)
    
    # Parse Premiere XML
    clips, total_duration = premiere_parser.parse_premiere_xml(args.input)
    
    # Convert to sequence
    sequence = premiere_parser.clips_to_track_sequence(clips, total_duration)
    
    # Save outputs
    output_dir = Path(args.output)
    
    if args.format in ['npz', 'both']:
        npz_path = output_dir / f'{video_id}_tracks.npz'
        premiere_parser.save_sequence(sequence, str(npz_path), video_id)
    
    if args.format in ['csv', 'both']:
        csv_path = output_dir / f'{video_id}_tracks.csv'
        premiere_parser.save_csv(sequence, str(csv_path), video_id, clips)
    
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
    logger.info(f"Unique assets: {len(premiere_parser.asset_mapping)}")
    logger.info("\nAsset Mapping:")
    for asset_name, asset_id in sorted(premiere_parser.asset_mapping.items(), key=lambda x: x[1]):
        logger.info(f"  {asset_id}: {asset_name}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
