# Mesh Focus Orbit

Blender 5.2 用の単体アドオンです。

## インストール

1. Blender の `Edit > Preferences > Add-ons > Install...` を開く
2. `mesh_focus_orbit.py` を選択する
3. アドオンを有効化する

## 使い方

まず Add-on Preferences の `Reference Object` に、リトポロジー用の参照ハイポリメッシュを指定してください。
3D Viewport の中央に対象メッシュを表示し、初期設定の `Right Shift` を短時間に2回押します。
2回目の押下で Reference Object だけへ画面中央から一度だけRay Castし、交点を一時Orbit中心にします。
右上の Navigation Gizmo または MMB で回転し、編集を行ってください。
もう一度 `Right Shift` を短時間に2回押すと一時モードを終了します。
解除時は、一時モード開始直前に保存したViewportの視点（位置・回転・距離・透視/正投影）へ戻します。
編集したメッシュや選択状態などの作業データは変更しません。

画面中央に Reference Object の表面がない場合、または Reference Object が未指定の場合は何も変更しません。

Reference Object の Face Set を使って局所作業範囲を一時表示したい場合は、`Ctrl` を押しながら同じ `Right Shift` を短時間に2回押します。
画面中央の Reference Object の Face Set ID を取得し、その ID の面だけを一時表示してから同じ一時Orbitを開始します。
リトポメッシュを Edit Mode のまま使用でき、Active Object、選択状態、Reference Object の `hide_select` は変更しません。
解除時は Reference Object の `.hide_poly` とViewport状態を開始前へ復元します。

Sculpt Mode でカーソル位置のFace Setを髪束単位で一発適用したい場合は、対象面にカーソルを置いたまま、3D View上で `E` を押します。カーソル直下のFaceをseedにし、Edgeを共有するFaceだけを辿りながら、面積重み付きの近傍平滑化法線、Raw dihedral、凹面ペナルティを使ったbottleneck型の領域成長を行います。強い谷を越えにくくし、滑らかに続く髪束を優先して、seedと同じ既存のFace Set IDを候補Faceへ直接適用します。プレビュー、Mask、`sculpt.expand`、Timerは使用しません。結果が気に入らない場合は通常の `Ctrl+Z` で1回の操作全体を戻してください。F3メニューからの起動はカーソル位置を正しく取得できないため対応していません。

`E` は Blender の `Preferences > Keymap` で `Mesh Focus: Local Face Set Grow` を検索して変更できます。

設定は `Edit > Preferences > Add-ons > Mesh Focus Orbit` にあります。

- `Enable`: アドオンの有効/無効
- `Activation Key`: ダブルタップに使う左右のCtrl / Shift / Altから選択（既定は `Right Shift`）
- `Reference Object`: 通常MFOとFace Set MFOがRay Castする参照ハイポリメッシュ。未指定時は両機能を起動しません
- `Focus Loss Behavior`: Blenderウィンドウがフォーカスを失ったときの動作。`Keep Mode`（維持、既定）または `Exit Mode`（解除）から選択
- `Double-tap Window`: 2回の押下を1回のトグルと判定する時間（秒）
- `Show Mode Indicator`: 一時モード中に3D Viewヘッダーへ `MESH FOCUS ORBIT ON` を表示（既定ON）
- `Debug Display`: 一時Orbit中心を青いポイントで表示（既定OFF）

通常MFOとFace Set MFOはReference Object以外をRay Cast対象にしません。選択状態、3D Cursor、Pivot Point、Object Transform、常設の Orbit Around Selection は変更しません。

新しい `.blend` ファイルを開くと、Viewportやメッシュへの参照を安全に破棄するため一時モードを解除します。
