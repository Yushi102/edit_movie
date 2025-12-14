"""Quick feature validation - validates a sample of videos"""
import numpy as np
import pandas as pd
from pathlib import Path
from feature_alignment import FeatureAligner

def quick_validate():
    """Quick validation of a few sample videos"""
    features_dir = Path("input_features")
    tracks_dir = Path("preprocessed_data/xml_tracks")
    aligner = FeatureAligner(tolerance=0.05)
    
    # Get all track files
    track_files = list(tracks_dir.glob("*_tracks.npz"))
    print(f"Found {len(track_files)} track files")
    
    # Sample 5 videos
    sample_files = track_files[:5]
    
    results = []
    for track_file in sample_files:
        video_name = track_file.stem.replace('_tracks', '')
        print(f"\nValidating: {video_name}")
        
        # Load track data
        track_data = np.load(track_file)
        sequences = track_data['sequences']
        fps = float(track_data['fps'])
        track_times = np.arange(sequences.shape[0]) / fps
        
        print(f"  Track: {len(track_times)} timesteps, {fps} FPS")
        
        # Check for audio features
        audio_file = features_dir / f"{video_name}_features.csv"
        audio_df = None
        if audio_file.exists():
            audio_df = pd.read_csv(audio_file)
            print(f"  Audio: {len(audio_df)} rows, {len(audio_df.columns)} columns")
        else:
            print(f"  Audio: NOT FOUND")
        
        # Check for visual features
        visual_file = features_dir / f"{video_name}_visual_features.csv"
        visual_df = None
        if visual_file.exists():
            visual_df = pd.read_csv(visual_file)
            print(f"  Visual: {len(visual_df)} rows, {len(visual_df.columns)} columns")
        else:
            print(f"  Visual: NOT FOUND")
        
        # Test alignment if features exist
        if audio_df is not None or visual_df is not None:
            try:
                aligned_audio, aligned_visual, modality_mask, stats = aligner.align_features(
                    track_times, audio_df, visual_df, video_id=video_name
                )
                print(f"  Alignment successful!")
                print(f"    Audio coverage: {stats.get('audio_coverage_pct', 0):.1f}%")
                print(f"    Visual coverage: {stats.get('visual_coverage_pct', 0):.1f}%")
                print(f"    Audio interpolation: {stats.get('audio_interpolated_pct', 0):.1f}%")
                print(f"    Visual interpolation: {stats.get('visual_interpolated_pct', 0):.1f}%")
                
                results.append({
                    'video': video_name,
                    'status': 'success',
                    'has_audio': audio_df is not None,
                    'has_visual': visual_df is not None,
                    'stats': stats
                })
            except Exception as e:
                print(f"  Alignment failed: {e}")
                results.append({
                    'video': video_name,
                    'status': 'failed',
                    'error': str(e)
                })
        else:
            results.append({
                'video': video_name,
                'status': 'missing_features'
            })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    successful = sum(1 for r in results if r['status'] == 'success')
    print(f"Successful: {successful}/{len(results)}")
    print(f"Missing features: {sum(1 for r in results if r['status'] == 'missing_features')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    
    return results

if __name__ == "__main__":
    quick_validate()
