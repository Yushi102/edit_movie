# セキュリティチェックリスト

リポジトリ公開前のセキュリティと個人情報のチェックリストです。

## ✅ 完了した対策

### 1. 個人情報の削除

- [x] **ユーザー名の削除**: `C:\Users\yushi\` → 汎用パスに変更
- [x] **個人的なパスの削除**: `D:\切り抜き\` → 汎用パスに変更
- [x] **ハードコードされたパスの削除**: すべてのスクリプトとドキュメントから削除

#### 修正したファイル:
- `src/data_preparation/xml2csv.py` - デフォルトパスを相対パスに変更
- `scripts/utilities/reextract_single_video.py` - サンプルパスをプレースホルダーに変更
- `test_fcpxml_extraction.bat` - 例示パスを汎用化
- `scripts/batch_processing/batch_test_fcpxml.bat` - 例示パスを汎用化
- `docs/guides/FCPXML_EXTRACTION_GUIDE.md` - ドキュメント内のパスを汎用化
- `docs/QUICK_START.md` - ドキュメント内のパスを汎用化

### 2. .gitignoreの強化

- [x] **動画ファイルの除外**: すべての動画形式（.mp4, .mov, .avi, .mkv）
- [x] **XMLファイルの除外**: 個人データを含む可能性のあるXMLファイル
- [x] **データディレクトリの除外**: `data/`ディレクトリ全体
- [x] **出力ファイルの除外**: `outputs/`ディレクトリ全体
- [x] **一時ファイルの除外**: `temp_features/`など

### 3. 機密情報のチェック

- [x] **APIキー**: なし（検索結果: 0件）
- [x] **パスワード**: なし（検索結果: 0件）
- [x] **トークン**: なし（検索結果: 0件）
- [x] **メールアドレス**: なし（検索結果: 0件）

## 📋 公開前の最終チェック

### 必須チェック項目

- [ ] **Gitの履歴確認**: 過去のコミットに個人情報が含まれていないか確認
- [ ] **大きなファイルの確認**: 動画ファイルやモデルファイルがコミットされていないか確認
- [ ] **README.mdの確認**: 個人情報や内部情報が含まれていないか確認
- [ ] **ライセンスファイルの追加**: `LICENSE`ファイルを追加（MIT Licenseなど）
- [ ] **CONTRIBUTINGガイドの追加**: 貢献方法を記載（オプション）

### 推奨チェック項目

- [ ] **サンプルデータの追加**: 小さなサンプル動画とXMLを追加（オプション）
- [ ] **CIの設定**: GitHub Actionsなどでテストを自動化（オプション）
- [ ] **ドキュメントの充実**: 使い方やトラブルシューティングを追加
- [ ] **デモ動画の作成**: 使い方を示すGIFやビデオを追加（オプション）

## 🔍 手動確認コマンド

### 1. 個人情報の検索

```bash
# ユーザー名の検索
git grep -i "yushi" -- ':!SECURITY_CHECKLIST.md'

# 個人的なパスの検索
git grep -E "C:\\\\Users\\\\|D:\\\\切り抜き" -- ':!SECURITY_CHECKLIST.md'

# メールアドレスの検索
git grep -E "\w+@\w+\.\w+"
```

### 2. 大きなファイルの確認

```bash
# 100KB以上のファイルを検索
find . -type f -size +100k -not -path "./.git/*" -not -path "./.venv/*"

# Gitで追跡されている大きなファイルを確認
git ls-files | xargs ls -lh | awk '$5 ~ /M$/ {print $9, $5}'
```

### 3. 機密情報の検索

```bash
# APIキー、パスワード、トークンの検索
git grep -iE "api_key|password|secret|token" -- ':!SECURITY_CHECKLIST.md'
```

## 🚨 緊急時の対応

### コミット済みの機密情報を削除する方法

もし機密情報をコミットしてしまった場合:

```bash
# 特定のファイルを履歴から完全に削除
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch path/to/sensitive/file" \
  --prune-empty --tag-name-filter cat -- --all

# または、BFG Repo-Cleanerを使用（推奨）
# https://rtyley.github.io/bfg-repo-cleaner/
bfg --delete-files sensitive_file.txt
git reflog expire --expire=now --all
git gc --prune=now --aggressive
```

**注意**: 履歴を書き換えた後は、強制プッシュが必要です:
```bash
git push origin --force --all
git push origin --force --tags
```

## 📝 公開後の注意事項

### 1. Issue/PRの監視

- 機密情報が含まれていないか確認
- スパムや不適切なコンテンツを削除

### 2. 定期的なセキュリティ監査

- 依存関係の脆弱性チェック: `pip audit`
- GitHub Security Alertsの確認

### 3. ライセンス遵守

- 使用しているライブラリのライセンスを確認
- 必要に応じてLICENSEファイルを更新

## 🔗 参考リンク

- [GitHub: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)
- [gitignore.io](https://www.toptal.com/developers/gitignore)

## ✅ 最終確認

公開前に以下を確認してください:

1. [ ] すべての個人情報が削除されている
2. [ ] .gitignoreが適切に設定されている
3. [ ] 大きなファイル（動画、モデル）がコミットされていない
4. [ ] READMEが適切に更新されている
5. [ ] LICENSEファイルが追加されている
6. [ ] 過去のコミット履歴に機密情報が含まれていない

すべてチェックが完了したら、安全に公開できます！

