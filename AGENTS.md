# Repository rule

配布対象のソース修正ごとに `bl_info` version のパッチ番号を必ずインクリメントし、変更とバージョン更新を同じ作業に含める。

# Blender add-on rules

- 実装・検証が実環境テスト候補まで完了したら、Luna実装担当が毎回Blenderへのインストールまで行う。親へのhandoffだけで完了としない。
- インストール、ライブ再読み込み、確認には `blender-official` MCPを優先し、利用可能な場合はComputer Useを使わない。
- 完成候補の `mesh_focus_orbit.py` は対象Blenderバージョンのユーザーアドオン配置先へ直接反映する。旧版バックアップはユーザーが明示的に要求した場合だけ作成する。
- `blender-official` 経由で対象モジュールを安全に無効化、再読み込み、再有効化し、Blender内の `bl_info['version']`、`addon_utils.check()`、実際のmodule file pathを確認して完了報告へ記載する。
- インストール確認では未保存シーンを保存せず、メッシュ、オブジェクト、シーン内容を変更しない。
- Blender未起動、MCP未接続、対象配置先不明などで安全にインストールできない場合は、推測やComputer Useへの自動フォールバックをせず、未実施理由を親へ報告する。
