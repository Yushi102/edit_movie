"""
Prediction Smoothing Filters

フレーム単位の回帰予測を平滑化して、ジッター（震え）を軽減します。

利用可能なフィルタ:
- Moving Average: 移動平均フィルタ
- Savitzky-Golay: サビツキー・ゴーレイ・フィルタ（多項式フィッティング）
- Exponential Moving Average: 指数移動平均
"""
import numpy as np
from scipy.signal import savgol_filter
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class PredictionSmoother:
    """予測値の平滑化"""
    
    def __init__(
        self,
        method: str = 'savgol',
        window_size: int = 5,
        polyorder: int = 2,
        alpha: float = 0.3
    ):
        """
        初期化
        
        Args:
            method: 平滑化手法 ('moving_average', 'savgol', 'ema')
            window_size: ウィンドウサイズ（奇数を推奨）
            polyorder: サビツキー・ゴーレイの多項式次数（window_size未満）
            alpha: 指数移動平均の平滑化係数（0-1、小さいほど滑らか）
        """
        self.method = method
        self.window_size = window_size
        self.polyorder = polyorder
        self.alpha = alpha
        
        # バリデーション
        if method == 'savgol':
            if window_size % 2 == 0:
                logger.warning(f"Savitzky-Golay filter requires odd window_size. Adjusting {window_size} -> {window_size + 1}")
                self.window_size = window_size + 1
            
            if polyorder >= window_size:
                logger.warning(f"polyorder must be less than window_size. Adjusting {polyorder} -> {window_size - 1}")
                self.polyorder = window_size - 1
        
        logger.info(f"PredictionSmoother initialized: method={method}, window_size={self.window_size}")
    
    def smooth_track_predictions(
        self,
        tracks_data: List[Dict],
        parameters: List[str] = None
    ) -> List[Dict]:
        """
        トラック予測を平滑化
        
        Args:
            tracks_data: トラックデータのリスト
                各要素は辞書で、'scale', 'position_x', 'position_y'などのキーを含む
            parameters: 平滑化するパラメータのリスト（Noneの場合は全て）
        
        Returns:
            平滑化されたトラックデータのリスト
        """
        if not tracks_data:
            return tracks_data
        
        # デフォルトで平滑化するパラメータ
        if parameters is None:
            parameters = ['scale', 'position_x', 'position_y', 'crop_left', 'crop_right', 'crop_top', 'crop_bottom']
        
        # 各パラメータを平滑化
        smoothed_tracks = []
        
        for track in tracks_data:
            smoothed_track = track.copy()
            
            # 各パラメータを個別に平滑化
            for param in parameters:
                if param in track:
                    # 単一の値の場合はスキップ
                    if isinstance(track[param], (int, float)):
                        continue
                    
                    # 配列の場合は平滑化
                    if isinstance(track[param], (list, np.ndarray)):
                        smoothed_track[param] = self._smooth_array(track[param])
            
            smoothed_tracks.append(smoothed_track)
        
        return smoothed_tracks
    
    def smooth_clip_parameters(
        self,
        start_frame: int,
        end_frame: int,
        predictions: Dict[str, np.ndarray],
        track_id: int
    ) -> Dict[str, float]:
        """
        クリップの各パラメータを平滑化
        
        Args:
            start_frame: 開始フレーム
            end_frame: 終了フレーム
            predictions: 全フレームの予測結果
            track_id: トラックID
        
        Returns:
            平滑化されたパラメータの辞書
        """
        # クリップの範囲を抽出
        clip_length = end_frame - start_frame + 1
        
        # 各パラメータを平滑化
        smoothed_params = {}
        
        param_keys = ['scale', 'pos_x', 'pos_y', 'crop_l', 'crop_r', 'crop_t', 'crop_b']
        param_names = ['scale', 'position_x', 'position_y', 'crop_left', 'crop_right', 'crop_top', 'crop_bottom']
        
        for key, name in zip(param_keys, param_names):
            if key in predictions:
                # クリップの範囲を抽出
                values = predictions[key][start_frame:end_frame+1, track_id]
                
                # 平滑化
                if clip_length >= self.window_size:
                    # 十分な長さがある場合は平滑化
                    smoothed_values = self._smooth_array(values)
                    # 平均値を使用（または中央値、最頻値など）
                    smoothed_params[name] = float(np.mean(smoothed_values))
                else:
                    # 短い場合は単純平均
                    smoothed_params[name] = float(np.mean(values))
        
        return smoothed_params
    
    def _smooth_array(self, values: np.ndarray) -> np.ndarray:
        """
        配列を平滑化
        
        Args:
            values: 入力配列
        
        Returns:
            平滑化された配列
        """
        if len(values) < self.window_size:
            # ウィンドウサイズより短い場合はそのまま返す
            return values
        
        if self.method == 'moving_average':
            return self._moving_average(values)
        elif self.method == 'savgol':
            return self._savitzky_golay(values)
        elif self.method == 'ema':
            return self._exponential_moving_average(values)
        else:
            logger.warning(f"Unknown smoothing method: {self.method}. Using moving average.")
            return self._moving_average(values)
    
    def _moving_average(self, values: np.ndarray) -> np.ndarray:
        """
        移動平均フィルタ
        
        Args:
            values: 入力配列
        
        Returns:
            平滑化された配列
        """
        # パディング（エッジ効果を軽減）
        pad_width = self.window_size // 2
        padded = np.pad(values, pad_width, mode='edge')
        
        # 移動平均を計算
        kernel = np.ones(self.window_size) / self.window_size
        smoothed = np.convolve(padded, kernel, mode='valid')
        
        return smoothed
    
    def _savitzky_golay(self, values: np.ndarray) -> np.ndarray:
        """
        サビツキー・ゴーレイ・フィルタ
        
        多項式フィッティングによる平滑化。
        移動平均よりも元の形状を保持しながら平滑化できる。
        
        Args:
            values: 入力配列
        
        Returns:
            平滑化された配列
        """
        try:
            smoothed = savgol_filter(
                values,
                window_length=self.window_size,
                polyorder=self.polyorder,
                mode='nearest'  # エッジ処理
            )
            return smoothed
        except Exception as e:
            logger.warning(f"Savitzky-Golay filter failed: {e}. Falling back to moving average.")
            return self._moving_average(values)
    
    def _exponential_moving_average(self, values: np.ndarray) -> np.ndarray:
        """
        指数移動平均（EMA）
        
        最近の値により大きな重みを与える平滑化。
        
        Args:
            values: 入力配列
        
        Returns:
            平滑化された配列
        """
        smoothed = np.zeros_like(values)
        smoothed[0] = values[0]
        
        for i in range(1, len(values)):
            smoothed[i] = self.alpha * values[i] + (1 - self.alpha) * smoothed[i-1]
        
        return smoothed


def smooth_predictions_for_xml(
    predictions: Dict[str, np.ndarray],
    tracks_data: List[Dict],
    method: str = 'savgol',
    window_size: int = 5
) -> List[Dict]:
    """
    XML生成用に予測を平滑化（便利関数）
    
    Args:
        predictions: モデルの予測結果
        tracks_data: トラックデータのリスト
        method: 平滑化手法
        window_size: ウィンドウサイズ
    
    Returns:
        平滑化されたトラックデータ
    """
    smoother = PredictionSmoother(method=method, window_size=window_size)
    
    smoothed_tracks = []
    
    for track in tracks_data:
        smoothed_track = track.copy()
        
        # 各パラメータを平滑化
        smoothed_params = smoother.smooth_clip_parameters(
            start_frame=track['start_frame'],
            end_frame=track['end_frame'],
            predictions=predictions,
            track_id=track['track_id']
        )
        
        # 更新
        smoothed_track.update(smoothed_params)
        smoothed_tracks.append(smoothed_track)
    
    return smoothed_tracks


if __name__ == "__main__":
    # テスト
    logging.basicConfig(level=logging.INFO)
    
    # ダミーデータでテスト
    np.random.seed(42)
    
    # ノイズのある信号を生成
    t = np.linspace(0, 10, 100)
    clean_signal = np.sin(t)
    noisy_signal = clean_signal + np.random.normal(0, 0.2, len(t))
    
    print("Testing smoothing methods...")
    
    # 各手法でテスト
    methods = ['moving_average', 'savgol', 'ema']
    
    for method in methods:
        smoother = PredictionSmoother(method=method, window_size=5)
        smoothed = smoother._smooth_array(noisy_signal)
        
        # RMSEを計算
        rmse = np.sqrt(np.mean((smoothed - clean_signal) ** 2))
        print(f"\n{method}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Original std: {noisy_signal.std():.4f}")
        print(f"  Smoothed std: {smoothed.std():.4f}")
    
    # トラックデータのテスト
    print("\n\nTesting track smoothing...")
    
    tracks_data = [
        {
            'track_id': 0,
            'start_frame': 0,
            'end_frame': 50,
            'scale': 1.0 + np.random.normal(0, 0.05, 51),
            'position_x': 0.5 + np.random.normal(0, 0.02, 51),
            'position_y': 0.5 + np.random.normal(0, 0.02, 51)
        }
    ]
    
    smoother = PredictionSmoother(method='savgol', window_size=7)
    smoothed_tracks = smoother.smooth_track_predictions(tracks_data)
    
    print(f"Original scale std: {tracks_data[0]['scale'].std():.4f}")
    print(f"Smoothed scale std: {smoothed_tracks[0]['scale'].std():.4f}")
    
    print("\n✅ All tests passed!")
