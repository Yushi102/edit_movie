"""
トレーニング用の特徴量を全動画から抽出（並列処理版）

XMLファイルから動画パスを読み取り、特徴量を並列抽出します。

並列化戦略:
- N_PARALLEL本を同時処理
- GPU処理（Demucs/CLIP/BERT）はプロセス間でロックして競合を防ぐ
- Whisper large-v3はCPU使用（GPUメモリを圧迫しない）
- RAM 48GBあるので2〜3並列は余裕
"""
import os
import sys
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote
from tqdm import tqdm
import logging
import multiprocessing
from multiprocessing import Pool, Manager

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 並列数（GPUメモリ競合を避けるため2に設定、余裕があれば3も可）
N_PARALLEL = 2


def extract_video_path_from_xml(xml_path: str) -> str:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        pathurl_elem = root.find('.//pathurl')
        if pathurl_elem is not None and pathurl_elem.text:
            path = pathurl_elem.text.replace('file://localhost/', '')
            path = unquote(path)
            path = path.replace('%3a', ':').replace('%3A', ':')
            path = path.replace('/', '\\')
            return path
    except Exception as e:
        logger.warning(f"XMLパース失敗 {xml_path}: {e}")
    return None


def worker(args):
    """並列ワーカー関数"""
    video_path, output_dir, gpu_lock = args

    # sys.pathを設定
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data_preparation.extract_video_features_parallel import extract_features_worker_with_lock
    return extract_features_worker_with_lock(video_path, output_dir, gpu_lock)


def main():
    xml_dir = "data/raw/editxml"
    output_dir = "data/processed/input_features"
    os.makedirs(output_dir, exist_ok=True)

    # XMLから動画パスを収集
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    logger.info(f"XMLファイル数: {len(xml_files)}")

    video_paths = []
    for xml_file in xml_files:
        video_path = extract_video_path_from_xml(xml_file)
        if video_path and os.path.exists(video_path):
            video_paths.append(video_path)
        elif video_path:
            logger.warning(f"動画ファイルが見つかりません: {video_path}")

    logger.info(f"有効な動画ファイル数: {len(video_paths)}")

    # 未処理のみ抽出
    to_process = []
    for video_path in video_paths:
        video_stem = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{video_stem}_features.csv")
        if os.path.exists(output_path):
            logger.info(f"スキップ（既に存在）: {video_stem}")
        else:
            to_process.append(video_path)

    logger.info(f"処理対象: {len(to_process)}個 / 並列数: {N_PARALLEL}")

    if not to_process:
        logger.info("全ての動画が既に処理済みです")
        return

    # GPUロック（プロセス間共有）
    manager = Manager()
    gpu_lock = manager.Lock()

    # 並列処理
    args_list = [(vp, output_dir, gpu_lock) for vp in to_process]

    success_count = 0
    error_count = 0

    logger.info("="*70)
    logger.info("特徴量抽出開始（並列処理）")
    logger.info("="*70)

    with Pool(processes=N_PARALLEL) as pool:
        for result in tqdm(pool.imap_unordered(worker, args_list), total=len(args_list), desc="抽出中"):
            if result['status'] == 'Success':
                success_count += 1
                logger.info(f"[OK] {result['file']}: {result['timesteps']}steps, {result['features']}features")
            else:
                error_count += 1
                logger.error(f"[ERROR] {result['file']}: {result.get('message', '不明')}")

    logger.info("="*70)
    logger.info(f"完了: 成功={success_count}, 失敗={error_count}")
    logger.info("="*70)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()
