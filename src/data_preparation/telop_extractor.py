"""
Telop (Graphics) Extractor from Premiere Pro XML

Extracts telop/graphics information from Premiere Pro XML files
and converts it to time-series data.
"""
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TelopExtractor:
    """Extract telop information from Premiere Pro XML"""
    
    def __init__(self, fps: float = 10.0):
        """
        Initialize TelopExtractor
        
        Args:
            fps: Frames per second for time-series sampling (default: 10.0)
        """
        self.fps = fps
        self.time_step = 1.0 / fps
        logger.info(f"TelopExtractor initialized: fps={fps}, time_step={self.time_step}s")
    
    def extract_telops_from_xml(self, xml_path: str) -> List[Dict]:
        """
        Extract telop information from XML
        
        Args:
            xml_path: Path to Premiere Pro XML file
        
        Returns:
            List of telop dictionaries with keys: text, start, end, duration
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            telops = []
            
            # Find all sequences
            for sequence in root.findall('.//sequence'):
                # Get timebase for time conversion
                timebase_elem = sequence.find('.//rate/timebase')
                timebase = float(timebase_elem.text) if timebase_elem is not None else 30.0
                
                # Find all video tracks
                for track in sequence.findall('.//video/track'):
                    # Find all clipitems in the track
                    for clipitem in track.findall('.//clipitem'):
                        # Check if this is a graphic
                        is_graphic = False
                        
                        # Check mediaSource
                        media_source = clipitem.find('.//mediaSource')
                        if media_source is not None and 'Graphic' in media_source.text:
                            is_graphic = True
                        
                        # Check effect category
                        for effect in clipitem.findall('.//effect'):
                            effect_category = effect.find('effectcategory')
                            if effect_category is not None and effect_category.text == 'graphic':
                                is_graphic = True
                                
                                # Get telop text from effect name
                                effect_name = effect.find('name')
                                if effect_name is not None and effect_name.text:
                                    telop_text = effect_name.text.strip()
                                    
                                    # Get timing information from clipitem
                                    start_elem = clipitem.find('start')
                                    end_elem = clipitem.find('end')
                                    
                                    if start_elem is not None and end_elem is not None:
                                        start_frames = int(start_elem.text)
                                        end_frames = int(end_elem.text)
                                        
                                        # Convert frames to seconds
                                        start_sec = start_frames / timebase
                                        end_sec = end_frames / timebase
                                        duration_sec = end_sec - start_sec
                                        
                                        telops.append({
                                            'text': telop_text,
                                            'start': start_sec,
                                            'end': end_sec,
                                            'duration': duration_sec
                                        })
                                        
                                        logger.debug(f"Found telop: '{telop_text}' at {start_sec:.2f}s - {end_sec:.2f}s")
            
            logger.info(f"Extracted {len(telops)} telops from {Path(xml_path).name}")
            return telops
            
        except Exception as e:
            logger.error(f"Failed to extract telops from {xml_path}: {e}")
            return []
    
    def telops_to_timeseries(
        self, 
        telops: List[Dict], 
        total_duration: float
    ) -> pd.DataFrame:
        """
        Convert telop list to time-series DataFrame
        
        Args:
            telops: List of telop dictionaries
            total_duration: Total duration of the video in seconds
        
        Returns:
            DataFrame with columns: time, telop_active, telop_text
        """
        # Create time points
        num_steps = int(np.ceil(total_duration / self.time_step))
        time_points = [round(i * self.time_step, 6) for i in range(num_steps + 1)]
        
        # Initialize arrays
        telop_active = np.zeros(len(time_points), dtype=int)
        telop_text = [''] * len(time_points)
        
        # Fill in telop information
        for telop in telops:
            text = telop['text']
            start = telop['start']
            end = telop['end']
            
            # Find time points within this telop's range
            for i, t in enumerate(time_points):
                if start <= t < end:
                    telop_active[i] = 1
                    # If multiple telops overlap, concatenate with separator
                    if telop_text[i]:
                        telop_text[i] += ' | ' + text
                    else:
                        telop_text[i] = text
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': time_points,
            'telop_active': telop_active,
            'telop_text': telop_text
        })
        
        # Replace empty strings with NaN for consistency
        df['telop_text'] = df['telop_text'].replace('', np.nan)
        
        logger.info(f"Created time-series with {len(df)} timesteps, {telop_active.sum()} active telop frames")
        
        return df
    
    def extract_and_convert(
        self, 
        xml_path: str, 
        total_duration: float
    ) -> pd.DataFrame:
        """
        Extract telops from XML and convert to time-series in one step
        
        Args:
            xml_path: Path to Premiere Pro XML file
            total_duration: Total duration of the video in seconds
        
        Returns:
            DataFrame with time-series telop data
        """
        telops = self.extract_telops_from_xml(xml_path)
        df = self.telops_to_timeseries(telops, total_duration)
        return df


if __name__ == "__main__":
    # Test the extractor
    logger.info("Testing TelopExtractor...")
    
    extractor = TelopExtractor(fps=10.0)
    
    # Test with sample XML
    xml_path = 'editxml/bandicam 2025-03-03 22-34-57-492.xml'
    
    # Extract telops
    telops = extractor.extract_telops_from_xml(xml_path)
    
    logger.info(f"\nExtracted {len(telops)} telops:")
    for i, telop in enumerate(telops[:10]):  # Show first 10
        logger.info(f"  {i+1}. '{telop['text']}' ({telop['start']:.2f}s - {telop['end']:.2f}s, {telop['duration']:.2f}s)")
    
    if len(telops) > 10:
        logger.info(f"  ... and {len(telops) - 10} more")
    
    # Convert to time-series (assuming 60 second video)
    if telops:
        total_duration = max(t['end'] for t in telops) + 1.0
        df = extractor.telops_to_timeseries(telops, total_duration)
        
        logger.info(f"\nTime-series DataFrame:")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {df.columns.tolist()}")
        logger.info(f"\nSample (first 20 rows with telops):")
        active_rows = df[df['telop_active'] == 1].head(20)
        logger.info(active_rows[['time', 'telop_active', 'telop_text']])
    
    logger.info("\n✅ TelopExtractor test complete!")
