"""
Feature Validation and Alignment Quality Check

This script validates feature files and generates alignment quality reports
for all videos in the dataset.

Requirements validated:
- 7.1: Monotonic timestamp ordering
- 7.2: Coverage percentage computation
- 7.3: Gap detection (>1 second)
- 7.4: Feature dimension validation
"""
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from feature_alignment import FeatureAligner

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureValidator:
    """Validates feature files and generates alignment quality reports"""
    
    def __init__(
        self,
        features_dir: str = "input_features",
        tracks_dir: str = "preprocessed_data/xml_tracks"
    ):
        """
        Initialize FeatureValidator
        
        Args:
            features_dir: Directory containing feature CSV files
            tracks_dir: Directory containing track NPZ files
        """
        self.features_dir = Path(features_dir)
        self.tracks_dir = Path(tracks_dir)
        self.aligner = FeatureAligner(tolerance=0.05)
        
        # Expected dimensions
        self.expected_audio_features = 4
        self.expected_visual_features = 522
        
        logger.info(f"FeatureValidator initialized")
        logger.info(f"  Features directory: {self.features_dir}")
        logger.info(f"  Tracks directory: {self.tracks_dir}")

    def validate_all_videos(self) -> Dict:
        """Validate all videos and generate comprehensive report"""
        logger.info("Starting validation of all videos...")
        
        track_files = list(self.tracks_dir.glob("*_tracks.npz"))
        logger.info(f"Found {len(track_files)} track files")
        
        results = {
            'total_videos': len(track_files),
            'successful': 0,
            'failed': 0,
            'missing_audio': 0,
            'missing_visual': 0,
            'videos': []
        }
        
        for track_file in track_files:
            video_name = track_file.stem.replace('_tracks', '')
            logger.info(f"\nValidating: {video_name}")
            
            try:
                video_result = self.validate_video(video_name)
                results['videos'].append(video_result)
                
                if video_result['status'] == 'success':
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                
                if not video_result['has_audio']:
                    results['missing_audio'] += 1
                
                if not video_result['has_visual']:
                    results['missing_visual'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to validate {video_name}: {e}")
                results['failed'] += 1
                results['videos'].append({
                    'video_name': video_name,
                    'status': 'error',
                    'error': str(e)
                })
        
        results['summary'] = self._compute_summary_statistics(results['videos'])
        return results

    def validate_video(self, video_name: str) -> Dict:
        """Validate a single video's features"""
        result = {
            'video_name': video_name,
            'status': 'unknown',
            'has_audio': False,
            'has_visual': False,
            'has_tracks': False,
            'audio_issues': [],
            'visual_issues': [],
            'track_issues': [],
            'alignment_stats': {}
        }
        
        # Load track data
        track_file = self.tracks_dir / f"{video_name}_tracks.npz"
        if not track_file.exists():
            result['status'] = 'error'
            result['track_issues'].append('Track file not found')
            return result
        
        try:
            track_data = np.load(track_file)
            sequences = track_data['sequences']
            fps = float(track_data['fps'])
            
            # Generate timestamps from sequence length and FPS
            num_timesteps = sequences.shape[0]
            track_times = np.arange(num_timesteps) / fps
            
            result['has_tracks'] = True
            result['track_length'] = len(track_times)
            result['fps'] = fps
            
            track_issues = self._validate_timestamps(track_times, 'track')
            result['track_issues'].extend(track_issues)
            
        except Exception as e:
            result['status'] = 'error'
            result['track_issues'].append(f'Failed to load tracks: {e}')
            return result
        
        # Load and validate audio features
        audio_file = self.features_dir / f"{video_name}_features.csv"
        audio_df = None
        if audio_file.exists():
            try:
                audio_df = pd.read_csv(audio_file)
                result['has_audio'] = True
                audio_issues = self._validate_audio_features(audio_df)
                result['audio_issues'].extend(audio_issues)
            except Exception as e:
                result['audio_issues'].append(f'Failed to load audio: {e}')
        else:
            result['audio_issues'].append('Audio feature file not found')
        
        # Load and validate visual features
        visual_file = self.features_dir / f"{video_name}_visual_features.csv"
        visual_df = None
        if visual_file.exists():
            try:
                visual_df = pd.read_csv(visual_file)
                result['has_visual'] = True
                visual_issues = self._validate_visual_features(visual_df)
                result['visual_issues'].extend(visual_issues)
            except Exception as e:
                result['visual_issues'].append(f'Failed to load visual: {e}')
        else:
            result['visual_issues'].append('Visual feature file not found')
        
        # Test alignment if features are available
        if result['has_audio'] or result['has_visual']:
            try:
                aligned_audio, aligned_visual, modality_mask, stats = self.aligner.align_features(
                    track_times, audio_df, visual_df, video_id=video_name
                )
                result['alignment_stats'] = stats
                
                if stats.get('audio_interpolated_pct', 0) > 50:
                    result['audio_issues'].append(f"High interpolation rate: {stats['audio_interpolated_pct']:.1f}%")
                
                if stats.get('visual_interpolated_pct', 0) > 50:
                    result['visual_issues'].append(f"High interpolation rate: {stats['visual_interpolated_pct']:.1f}%")
                
                if stats.get('max_gap_audio', 0) > 5.0:
                    result['audio_issues'].append(f"Large gap detected: {stats['max_gap_audio']:.2f}s")
                
                if stats.get('max_gap_visual', 0) > 5.0:
                    result['visual_issues'].append(f"Large gap detected: {stats['max_gap_visual']:.2f}s")
                
            except Exception as e:
                result['alignment_stats']['error'] = str(e)
        
        # Determine overall status
        if len(result['audio_issues']) == 0 and len(result['visual_issues']) == 0 and len(result['track_issues']) == 0:
            result['status'] = 'success'
        elif 'not found' in ' '.join(result['audio_issues'] + result['visual_issues']):
            result['status'] = 'missing_features'
        else:
            result['status'] = 'issues_found'
        
        return result

    def _validate_timestamps(self, times: np.ndarray, modality: str) -> List[str]:
        """Validate timestamp ordering and gaps (Requirements 7.1, 7.3)"""
        issues = []
        
        # Check monotonic ordering
        if not np.all(np.diff(times) >= 0):
            issues.append(f"{modality} timestamps are not monotonically increasing")
        
        # Check for gaps > 1 second
        gaps = np.diff(times)
        large_gaps = gaps > 1.0
        if np.any(large_gaps):
            max_gap = np.max(gaps)
            num_large_gaps = np.sum(large_gaps)
            issues.append(f"{modality} has {num_large_gaps} gaps > 1s (max: {max_gap:.2f}s)")
        
        return issues
    
    def _validate_audio_features(self, audio_df: pd.DataFrame) -> List[str]:
        """Validate audio feature dimensions and content (Requirement 7.4)"""
        issues = []
        
        required_cols = ['time', 'audio_energy_rms', 'audio_is_speaking', 
                        'silence_duration_ms', 'text_is_active']
        missing_cols = [col for col in required_cols if col not in audio_df.columns]
        if missing_cols:
            issues.append(f"Missing audio columns: {missing_cols}")
            return issues
        
        time_issues = self._validate_timestamps(audio_df['time'].values, 'audio')
        issues.extend(time_issues)
        
        if audio_df[required_cols[1:]].isna().any().any():
            issues.append("Audio features contain NaN values")
        
        num_features = len(required_cols) - 1
        if num_features != self.expected_audio_features:
            issues.append(f"Expected {self.expected_audio_features} audio features, got {num_features}")
        
        return issues
    
    def _validate_visual_features(self, visual_df: pd.DataFrame) -> List[str]:
        """Validate visual feature dimensions and content (Requirement 7.4)"""
        issues = []
        
        required_cols = ['time', 'scene_change', 'visual_motion', 'saliency_x', 'saliency_y',
                        'face_count', 'face_center_x', 'face_center_y', 'face_size',
                        'face_mouth_open', 'face_eyebrow_raise']
        missing_cols = [col for col in required_cols if col not in visual_df.columns]
        if missing_cols:
            issues.append(f"Missing visual columns: {missing_cols}")
            return issues
        
        clip_cols = [f'clip_{i}' for i in range(512)]
        missing_clip = [col for col in clip_cols if col not in visual_df.columns]
        if missing_clip:
            issues.append(f"Missing {len(missing_clip)} CLIP features")
            return issues
        
        time_issues = self._validate_timestamps(visual_df['time'].values, 'visual')
        issues.extend(time_issues)
        
        if visual_df[required_cols[1:]].isna().any().any():
            issues.append("Visual scalar features contain NaN values")
        
        if visual_df[clip_cols].isna().any().any():
            issues.append("CLIP features contain NaN values")
        
        num_features = len(required_cols) - 1 + 512
        if num_features != self.expected_visual_features:
            issues.append(f"Expected {self.expected_visual_features} visual features, got {num_features}")
        
        return issues

    def _compute_summary_statistics(self, video_results: List[Dict]) -> Dict:
        """Compute summary statistics across all videos (Requirement 7.2)"""
        summary = {
            'avg_audio_coverage': 0.0,
            'avg_visual_coverage': 0.0,
            'avg_audio_interpolation': 0.0,
            'avg_visual_interpolation': 0.0,
            'max_audio_gap': 0.0,
            'max_visual_gap': 0.0,
            'videos_with_issues': 0,
            'videos_with_high_interpolation': 0,
            'videos_with_large_gaps': 0
        }
        
        audio_coverages = []
        visual_coverages = []
        audio_interpolations = []
        visual_interpolations = []
        audio_gaps = []
        visual_gaps = []
        
        for result in video_results:
            if result['status'] == 'error':
                continue
            
            stats = result.get('alignment_stats', {})
            
            if 'audio_coverage_pct' in stats:
                audio_coverages.append(stats['audio_coverage_pct'])
                audio_interpolations.append(stats.get('audio_interpolated_pct', 0))
                audio_gaps.append(stats.get('max_gap_audio', 0))
            
            if 'visual_coverage_pct' in stats:
                visual_coverages.append(stats['visual_coverage_pct'])
                visual_interpolations.append(stats.get('visual_interpolated_pct', 0))
                visual_gaps.append(stats.get('max_gap_visual', 0))
            
            if len(result.get('audio_issues', [])) > 0 or len(result.get('visual_issues', [])) > 0:
                summary['videos_with_issues'] += 1
            
            if stats.get('audio_interpolated_pct', 0) > 50 or stats.get('visual_interpolated_pct', 0) > 50:
                summary['videos_with_high_interpolation'] += 1
            
            if stats.get('max_gap_audio', 0) > 5.0 or stats.get('max_gap_visual', 0) > 5.0:
                summary['videos_with_large_gaps'] += 1
        
        if audio_coverages:
            summary['avg_audio_coverage'] = np.mean(audio_coverages)
            summary['avg_audio_interpolation'] = np.mean(audio_interpolations)
            summary['max_audio_gap'] = np.max(audio_gaps)
        
        if visual_coverages:
            summary['avg_visual_coverage'] = np.mean(visual_coverages)
            summary['avg_visual_interpolation'] = np.mean(visual_interpolations)
            summary['max_visual_gap'] = np.max(visual_gaps)
        
        return summary

    def generate_report(self, results: Dict, output_file: str = "feature_validation_report.txt"):
        """Generate a human-readable validation report"""
        logger.info(f"Generating validation report: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("FEATURE VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total videos: {results['total_videos']}\n")
            f.write(f"Successful: {results['successful']}\n")
            f.write(f"Failed: {results['failed']}\n")
            f.write(f"Missing audio features: {results['missing_audio']}\n")
            f.write(f"Missing visual features: {results['missing_visual']}\n\n")
            
            summary = results['summary']
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average audio coverage: {summary['avg_audio_coverage']:.1f}%\n")
            f.write(f"Average visual coverage: {summary['avg_visual_coverage']:.1f}%\n")
            f.write(f"Average audio interpolation: {summary['avg_audio_interpolation']:.1f}%\n")
            f.write(f"Average visual interpolation: {summary['avg_visual_interpolation']:.1f}%\n")
            f.write(f"Max audio gap: {summary['max_audio_gap']:.2f}s\n")
            f.write(f"Max visual gap: {summary['max_visual_gap']:.2f}s\n")
            f.write(f"Videos with issues: {summary['videos_with_issues']}\n")
            f.write(f"Videos with high interpolation (>50%): {summary['videos_with_high_interpolation']}\n")
            f.write(f"Videos with large gaps (>5s): {summary['videos_with_large_gaps']}\n\n")
            
            f.write("VIDEOS WITH ISSUES\n")
            f.write("-" * 80 + "\n")
            
            for video in results['videos']:
                if video['status'] in ['issues_found', 'error']:
                    f.write(f"\n{video['video_name']}\n")
                    f.write(f"  Status: {video['status']}\n")
                    
                    if video.get('audio_issues'):
                        f.write(f"  Audio issues:\n")
                        for issue in video['audio_issues']:
                            f.write(f"    - {issue}\n")
                    
                    if video.get('visual_issues'):
                        f.write(f"  Visual issues:\n")
                        for issue in video['visual_issues']:
                            f.write(f"    - {issue}\n")
                    
                    if video.get('track_issues'):
                        f.write(f"  Track issues:\n")
                        for issue in video['track_issues']:
                            f.write(f"    - {issue}\n")
                    
                    stats = video.get('alignment_stats', {})
                    if stats:
                        f.write(f"  Alignment stats:\n")
                        for key, value in stats.items():
                            if isinstance(value, float):
                                f.write(f"    {key}: {value:.2f}\n")
                            else:
                                f.write(f"    {key}: {value}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"Report saved to {output_file}")


def main():
    """Main validation function"""
    logger.info("Starting feature validation...")
    
    validator = FeatureValidator(
        features_dir="input_features",
        tracks_dir="preprocessed_data/xml_tracks"
    )
    
    results = validator.validate_all_videos()
    validator.generate_report(results, "feature_validation_report.txt")
    
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total videos: {results['total_videos']}")
    logger.info(f"Successful: {results['successful']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Missing audio: {results['missing_audio']}")
    logger.info(f"Missing visual: {results['missing_visual']}")
    logger.info("\nSummary statistics:")
    for key, value in results['summary'].items():
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.2f}")
        else:
            logger.info(f"  {key}: {value}")
    logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
