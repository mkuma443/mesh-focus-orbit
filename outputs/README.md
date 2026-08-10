# Temporary Orbit Around View Center

Blender 5.2 用の単体アドオンです。

## インストール

1. Blender の `Edit > Preferences > Add-ons > Install...` を開く
2. `temporary_orbit_around_view_center.py` を選択する
3. アドオンを有効化する

## 使い方

3D Viewport の中央に対象メッシュを表示し、初期設定の `Right Shift` を短時間に2回押します。
2回目の押下で画面中央から一度だけRay Castし、最前面の可視メッシュ交点を一時Orbit中心にします。
右上の Navigation Gizmo または MMB で回転し、編集を行ってください。
もう一度 `Right Shift` を短時間に2回押すと一時モードを終了します。
解除時は、一時モード開始直前に保存したViewportの視点（位置・回転・距離・透視/正投影）へ戻します。
編集したメッシュや選択状態などの作業データは変更しません。

画面中央に可視メッシュがない場合は何も変更しません。

設定は `Edit > Preferences > Add-ons > Temporary Orbit Around View Center` にあります。

- `Enable`: アドオンの有効/無効
- `Activation Key`: ダブルタップに使う左右のCtrl / Shift / Altから選択（既定は `Right Shift`）
- `Double-tap Window`: 2回の押下を1回のトグルと判定する時間（秒）
- `Show Mode Indicator`: 一時モード中に3D Viewヘッダーへ `TEMPORARY ORBIT ON` を表示（既定ON）
- `Debug Display`: 一時Orbit中心を青いポイントで表示（既定OFF）

Edit Mode ではライブの編集メッシュをRay Castします。選択状態、3D Cursor、Pivot Point、Object Transform、常設の Orbit Around Selection は変更しません。
