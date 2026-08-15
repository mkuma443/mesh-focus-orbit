# Mesh Focus Orbit

Blender 5.2 用のリトポロジー支援アドオンです。

現在のアドオンバージョン: **3.0.1**

主な機能:

- **通常 MFO**: 画面中央の Reference Object 表面を一時 Orbit 中心にする
- **Face Set MFO (FSMFO)**: Reference Object の対象 Face Set と、対応する Retopo island だけを一時表示する
- **Smart Face Set Fill**: Sculpt Mode でカーソル直下の髪束領域を一発で Face Set 化する

## インストール

1. Blender の `Edit > Preferences > Add-ons > Install...` を開く
2. `mesh_focus_orbit.py` を選択する
3. `Mesh Focus Orbit` を有効にする
4. Add-on Preferences の `Reference Object` に参照ハイポリメッシュを指定する

Reference Object は、リトポロジー対象のハイポリメッシュです。通常 MFO と FSMFO はこのオブジェクトだけを Ray Cast 対象にします。Retopo mesh やその他の Scene mesh は Orbit 中心判定の対象になりません。

## 基本操作

初期設定の Activation Key は `Right Shift` です。

| 機能 | 操作 |
| --- | --- |
| 通常 MFO ON/OFF | 設定キーを短時間に2回押す |
| Face Set MFO ON/OFF | `Ctrl` を押しながら設定キーを短時間に2回押す |
| Smart Face Set Fill | Sculpt Mode で `E` |
| Smart Face Set Fill 厳格モード | Sculpt Mode で `Shift + E` |

通常 MFO と FSMFO は別の KeyMap Item から直接起動します。FSMFO は外側の非 Undo Trigger を経由せず、Undo 対象の Activation Operator が直接起動し、その中から非 Undo Watcher を開始します。

## 通常 MFO

1. Reference Object を画面中央に置く
2. 設定キーをダブルタップする
3. 画面中央のスクリーン座標から Reference Object だけへ Ray Cast する
4. 最初の交点を一時的な Orbit 中心にする
5. Navigation Gizmo または MMB で回転する
6. もう一度設定キーをダブルタップして解除する

マウスカーソル位置ではなく、常に 3D Viewport の画面中央を使用します。Activation 時に一度だけ交点を計算し、MFO 中は中心を再計算しません。

解除時は、開始前に保存した `view_location`、`view_distance`、`view_rotation`、`view_perspective` を復元します。メッシュ、選択状態、3D Cursor、Pivot Point、Object Transform、常設の Orbit Around Selection 設定は変更しません。

右上には通常 `MESH FOCUS ORBIT ON` が表示されます。表示位置や表示自体は Preferences で変更できます。

## Face Set MFO

FSMFO は、Reference Object に `.sculpt_face_set` が存在する場合に使用できます。

1. 作業対象を画面中央に置く
2. `Ctrl + 設定キー`をダブルタップする
3. Reference Object の中央 Ray Hit から Face Set ID を取得する
4. 対象 Face Set だけを表示する一時 Proxy を生成する
5. 元の Reference Object の Viewport 描画を一時的に隠す
6. 必要に応じて Retopo_Work の対象 island だけを表示する
7. Proxy を対象に Orbit する
8. `Ctrl + 設定キー`を再度ダブルタップして解除する

Reference Object は Active Object に変更せず、Retopo_Work は Edit Mode のまま維持します。Reference Object は Shrinkwrap Target として元のまま残り、Proxy は Shrinkwrap Target には使用しません。

### Retopo island isolation

Retopo_Work の Face Set 対応付けは行いません。現在の Retopo mesh の edge-connected component を island として扱います。

FSMFO 起動時、各 island の頂点を Face Set Proxy の BVH へ最近傍投影し、次の複合指標で現在の作業 island を自動判定します。

- `near_ratio`
- median distance
- p90 distance

Face 数ではなく、island の大部分が Proxy に沿っているかを重視します。判定が十分明確でない場合は、Retopo isolation を無理に行いません。対象外の既存 Face、Edge、Vertex は一時的に非表示になります。FSMFO 中に作成された新規 geometry は通常どおり編集できます。

解除時には、開始前の hide 状態を復元します。FSMFO 中に作成された新規要素は表示状態を維持します。

### RetopoFlow compatibility

Preferences の `RetopoFlow Focus-Island Snap/Weld Filter` は既定で OFF です。

ON にした場合、FSMFO 中の PolyPen Snap と Translate Auto Merge に限定して、現在の target island に属すると確認できる既存頂点だけを候補にします。RetopoFlow の標準条件は維持し、FSMFO 以外の RetopoFlow 処理には介入しません。FSMFO 終了時には一時 hook を解除します。

### Undo

FSMFO の Activation は独立した Undo Operator として完了し、Watcher は非 Undo の監視処理として動作します。

- FSMFO 中の通常編集を Undo: 編集操作を戻し、FSMFO は維持する
- Activation 自体を Undo: FSMFO を終了し、Proxy、Reference 表示、Retopo isolation を元へ戻す

Blender 本体の Access Violation を避けるため、Undo の自動実行や自動テストは行いません。

## Smart Face Set Fill

Sculpt Mode でカーソルを Face Set 上へ置き、`E` を押すと一発適用します。マウスドラッグによるプレビューや Timer は使用しません。

処理は次のように動作します。

- カーソル直下の Face を seed にする
- seed と同じ既存 Face Set ID を使用する
- 共有 Edge だけを辿る
- 面積重み付きの平滑化法線で細かい normal ノイズを抑える
- raw dihedral と凹面ペナルティで髪束間の谷を越えにくくする
- bottleneck 型の遷移コストで、長く滑らかな髪束を優先する
- 対象 Face に既存の seed Face Set ID を直接書き込む

`Shift + E` は厳格モードです。通常モードより短い探索範囲と厳しい境界判定を使います。結果が気に入らない場合は、通常の `Ctrl + Z` で操作全体を1ステップ戻してください。

この機能は Sculpt Mode 専用です。`bpy.ops.sculpt.expand()`、Sculpt Mask、GPU Preview、Timer、Face Set の新規 ID 生成は使用しません。

ショートカットは Blender の `Preferences > Keymap` で `Mesh Focus: Local Face Set Grow` を検索して変更できます。

## Preferences

`Edit > Preferences > Add-ons > Mesh Focus Orbit` にあります。

- `Enable`: アドオンの有効/無効
- `Activation Key`: 通常 MFO と FSMFO のダブルタップキー。左右の Ctrl / Shift / Alt を選択可能
- `Reference Object`: 通常 MFO と FSMFO が Ray Cast する参照ハイポリメッシュ
- `Focus Loss Behavior`: Blender がフォーカスを失ったときにモードを維持するか解除するか
- `Double-tap Window`: ダブルタップと判定する時間幅
- `Show Mode Indicator`: MFO/FSMFO の状態表示
- `Debug Display`: Orbit 中心のデバッグポイント表示
- `RetopoFlow Focus-Island Snap/Weld Filter`: FSMFO 中の RetopoFlow Snap/Weld 制限。既定 OFF

## 制限と復旧

- Reference Object が未指定の場合、通常 MFO と FSMFO は起動しません
- 画面中央に Reference Object の表面がない場合は起動しません
- FSMFO には Reference Object の `.sculpt_face_set` が必要です
- RetopoFlow と PolyQuilt の連携機能は、それぞれのアドオンがインストールされている場合だけ有効になります
- ファイルロード、アドオン無効化、ウィンドウ終了時には一時 Proxy、isolation、hook をクリーンアップします

---

# Mesh Focus Orbit — English

A Blender 5.2 add-on for manual retopology workflows.

Current add-on version: **3.0.1**

Main features:

- **Normal MFO**: temporarily orbits around the surface point at the center of the viewport
- **Face Set MFO (FSMFO)**: temporarily shows one Reference Face Set and its matching Retopo island
- **Smart Face Set Fill**: applies the Face Set under the cursor to a geometry-aware hair-bundle region in one step

## Installation

1. Open `Edit > Preferences > Add-ons > Install...` in Blender
2. Select `mesh_focus_orbit.py`
3. Enable `Mesh Focus Orbit`
4. Set the retopology high-poly mesh in the add-on's `Reference Object` field

The Reference Object is the high-poly mesh used for retopology. Normal MFO and FSMFO ray-cast only this explicitly configured object. Retopo meshes and other scene meshes are not considered for the orbit-center ray cast.

## Controls

The default Activation Key is `Right Shift`.

| Feature | Shortcut |
| --- | --- |
| Normal MFO ON/OFF | Double-tap the configured key |
| Face Set MFO ON/OFF | Hold `Ctrl` and double-tap the configured key |
| Smart Face Set Fill | `E` in Sculpt Mode |
| Strict Smart Face Set Fill | `Shift + E` in Sculpt Mode |

Normal MFO and FSMFO use separate KeyMap Items. FSMFO is started directly by its Undo-enabled Activation Operator; it does not pass through an outer non-Undo trigger. The Activation Operator starts the non-Undo Watcher and then finishes.

## Normal MFO

1. Place the Reference Object at the center of the viewport
2. Double-tap the configured key
3. Cast one ray from the viewport center to the Reference Object
4. Use the first hit point as the temporary orbit center
5. Orbit with the Navigation Gizmo or MMB
6. Double-tap the configured key again to exit

The mouse cursor position is not used for the hit test. The hit is calculated once on activation and is not recomputed while the mode is active.

On exit, the starting `view_location`, `view_distance`, `view_rotation`, and `view_perspective` are restored. Mesh data, selection, the 3D Cursor, Pivot Point, Object Transform, and the persistent Orbit Around Selection setting are not changed.

The viewport normally shows `MESH FOCUS ORBIT ON`. Its visibility can be changed in Preferences.

## Face Set MFO

FSMFO requires a `.sculpt_face_set` attribute on the Reference Object.

1. Place the work area at the center of the viewport
2. Hold `Ctrl` and double-tap the configured key
3. Read the Face Set ID from the center ray hit on the Reference Object
4. Generate a temporary Proxy showing only that Face Set
5. Temporarily hide the original Reference Object's viewport drawing
6. Optionally isolate the matching Retopo island
7. Orbit around the temporary hit point
8. Hold `Ctrl` and double-tap the configured key again to exit

The Reference Object is not made active, and Retopo_Work remains in Edit Mode. The original Reference Object remains the Shrinkwrap Target; the Proxy is never used as the Shrinkwrap Target.

### Retopo island isolation

Retopo faces are not mapped to Reference Face Set IDs. Each edge-connected component of the current Retopo mesh is treated as an island.

At FSMFO activation, each island is evaluated by nearest-point distances from its vertices to a BVH built from the Face Set Proxy. The composite confidence uses:

- `near_ratio`
- median distance
- p90 distance

The decision emphasizes whether most of an island follows the Proxy, rather than the island's face count. If the best candidate is not sufficiently clear, Retopo isolation is skipped. Existing non-target faces, edges, and vertices are temporarily hidden. Geometry created during FSMFO remains available for editing.

The original hide state is restored on exit. New elements created during FSMFO remain visible.

### RetopoFlow compatibility

`RetopoFlow Focus-Island Snap/Weld Filter` is OFF by default.

When enabled, the filter is limited to PolyPen Snap and Translate Auto Merge while FSMFO is active. Only existing vertices confirmed to belong to the current target island are allowed as candidates. RetopoFlow's standard conditions are preserved, and other RetopoFlow paths are left untouched. Temporary hooks are released when FSMFO ends.

### Undo

FSMFO Activation is a separate Undo-enabled operator that finishes immediately. The Watcher is a non-Undo monitor.

- Undoing an edit made inside FSMFO: the edit is undone and FSMFO remains active
- Undoing the FSMFO Activation: FSMFO ends and the Proxy, Reference visibility, and Retopo isolation are restored

Undo is not executed automatically. This avoids repeating Blender Access Violation scenarios during automated tests.

## Smart Face Set Fill

In Sculpt Mode, place the cursor over a Face Set and press `E` to apply it in one operation. There is no drag preview or Timer.

The algorithm:

- Uses the face under the cursor as the seed
- Reuses the seed's existing Face Set ID
- Traverses only edge-connected faces
- Smooths local normals with area weighting to reduce dense-mesh noise
- Uses raw dihedral and concavity penalties to resist valleys between hair bundles
- Uses a bottleneck-style transition cost to favor long, smooth bundles
- Writes the existing seed Face Set ID directly to the accepted faces

`Shift + E` enables Strict Mode with a shorter search range and stricter boundary decisions. Use normal Blender `Ctrl + Z` to undo the complete operation in one step.

This feature is Sculpt Mode only. It does not use `bpy.ops.sculpt.expand()`, Sculpt Mask, GPU preview, Timer-driven interaction, or newly generated Face Set IDs.

The shortcut can be changed in Blender's `Preferences > Keymap` by searching for `Mesh Focus: Local Face Set Grow`.

## Preferences

Open `Edit > Preferences > Add-ons > Mesh Focus Orbit`.

- `Enable`: Enable or disable the add-on
- `Activation Key`: The double-tap key for Normal MFO and FSMFO; left/right Ctrl, Shift, and Alt are available
- `Reference Object`: The high-poly object used by Normal MFO and FSMFO ray casts
- `Focus Loss Behavior`: Keep or exit the mode when Blender loses focus
- `Double-tap Window`: Time window used to recognize a double-tap
- `Show Mode Indicator`: Show the MFO/FSMFO status indicator
- `Debug Display`: Show the temporary orbit-center debug point
- `RetopoFlow Focus-Island Snap/Weld Filter`: Restrict RetopoFlow Snap/Weld candidates during FSMFO; OFF by default

## Limitations and recovery

- Normal MFO and FSMFO do not start without a configured Reference Object
- They do not start when the viewport-center ray misses the Reference Object
- FSMFO requires the Reference Object's `.sculpt_face_set` attribute
- RetopoFlow and PolyQuilt integration is enabled only when the corresponding add-ons are installed
- Temporary Proxies, isolation state, and hooks are cleaned up during file loading, add-on disable, and window teardown
