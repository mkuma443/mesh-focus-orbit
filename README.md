# Mesh Focus Orbit

Blender 5.2 用のリトポロジー支援アドオンです。

現在のアドオンバージョン: **3.2.24**

主な機能:

- **通常 MFO**: Object / Edit / Sculpt Mode で、画面中央の Reference Object 表面を一時 Orbit 中心にする
- **Face Set MFO (FSMFO)**: Reference Object の対象 Face Set と、対応する Retopo island だけを一時表示する
- **Smart Face Set Fill**: Sculpt Mode でカーソル直下の外周境界だけをプレビューし、E再押下またはEnterで外周内のseed接続面を Face Set 化する
- Smart Face Set Fill の予測は境界 edge の線だけを描画する。確定時に世代固定したcompact graphの外周境界を越えず、seed接続面を一括で復元して書き込む
- **Topology Colors**: 編集中の選択面へ6色の半透明ガイドを割り当てる

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
| 通常 MFO ON/OFF | Object / Edit / Sculpt Mode で設定キーを短時間に2回押す |
| Face Set MFO ON/OFF | Object / Edit Mode で `Ctrl` を押しながら設定キーを短時間に2回押す |
| Smart Face Set Fill プレビュー | Sculpt Mode で `E`、開始Eのrelease後に `E` を再押下（または `Enter`）で確定、ホイールで距離変更、`Esc` で取消 |
| Smart Face Set Fill 厳格プレビュー | Sculpt Mode で `Ctrl + E`、開始Eのrelease後に `E` を再押下（または `Enter`）で確定、ホイールで距離変更、`Esc` で取消 |
| Topology Colors | 編集モードで `Ctrl + Alt + 1..6`（`0`で解除） |

通常 MFO と FSMFO は別の KeyMap Item から直接起動します。FSMFO は外側の非 Undo Trigger を経由せず、Undo 対象の Activation Operator が直接起動し、その中から非 Undo Watcher を開始します。

## 通常 MFO

1. Reference Object を画面中央に置く（Object / Edit / Sculpt Mode で利用できます）
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

Sculpt Mode でカーソルを Face Set 上へ置き、`E` を押すと局所プレビューを開始します。開始EをreleaseしてからEを再押下、または `Enter` で確定し、`Esc` で取消します。開始キーの押しっぱなしやrepeat、ready前のE/Enterでは確定しません。プレビュー中の左/右クリックは確定・取消に使わず、Sculptのstrokeや選択へ渡しません。

処理は次のように動作します。

- カーソル直下の Face を seed にする
- seed と同じ既存 Face Set ID を使用する
- 共有 Edge と、両端点・向きが一致する未結合の継ぎ目を辿る（メッシュは結合しない）
- 平滑化した法線と複数の範囲の曲率から、段差・谷の境界を検出する
- 凸の頂上は通過し、方向別の凹み判定で付け根の谷を検出する。境界は1近傍分広げる
- 滑らかな領域を取得し、境界の帯は近い領域へ割り当てる。帯から別領域へ再拡張しない
- 非表示面・非多様体エッジは横断しない。rayが非表示面に当たった場合はその面を越えて最初の可視面をseedにする。10万面の固定上限は設けず、探索距離は初期半径の16倍を上限にホイールで調整する
- 対象 Face に既存の seed Face Set ID を直接書き込む
- 固定した複数方向の Lambert 照度近似を使い、まず指定距離の局所 patch を取得してから谷の符号付き変化を検出する。実影や AO、画面の照明は使わず、谷barrierでseed側の元face成分を選び、その周囲を元の面で補正する。半径変更ごとに現在patchから再判定するため、有限長の谷端での正当な再加入を妨げない
- 指定距離内の可視面graphで候補に囲まれた未選択成分だけを補完する。距離境界、欠落edge（crop/メッシュ境界、非多様体、hidden、未接続seam）へつながる成分や別sheetは補完しない
- 現在半径の候補距離内にある有効な共有辺crossingはshape/領域境界（orange）として扱い、fine partition barrierの有無で距離境界（cyan）へ戻さない。距離外または未探索のcrossingは距離境界のままにする
- 外周境界は現在のcompact radius domainから作り、内部閉ループを外部到達ラベルで除外する。外へ開く細い溝、crop/mesh境界、非多様体edge、hidden隣接、未接続seam、別sheetは外周側として保持する
- 境界分類では選択側ではなく隣接する非選択側の距離を使い、seed floodと最終描画のoutside面を一致させる
- 予測は外周境界線だけを表示し、面のtriangulationや面GPU batchを作らない。E再押下またはEnterで世代固定snapshotをseedからfloodし、その面集合を一度だけFace Setへ書き込む。履歴は現在と直前2段階を保持する
- 同期計算中と最新結果の実描画前後に届いたwheel入力は捨て、最新結果の描画後に短い排出区間を経た次のwheelだけを1段階として受け付ける。Escとready済み結果のE/Enter確定は維持する

初回準備では Edit Mode へ切り替えず、全 Face の center と全 loop の edge 次数だけを読み取ります。そこからカーソル seed の Euclidean 範囲と解析用 halo を切り出し、範囲内の polygon の edge、法線、非表示状態だけで局所 CSR、距離、谷・境界判定を作ります。全体の曲率や partition を先に作ってから切り出す処理は行いません。edge の全体次数が2の共有だけを通常接続とし、次数1の継ぎ目は両端点の一致と逆向き、法線の互換性を確認した場合だけ橋渡しします。局所切断境界や非多様体 edge は継ぎ目として扱いません。ホイールで範囲を広げた場合は保持済み配列・面 record を使いながら要求された局所 crop と record を再構築し、縮小と同じ半径への再訪では準備済みの結果を再利用します。

`Ctrl + E` は厳格モードです。探索範囲は通常モードと同じで、境界判定だけを厳しくします。結果が気に入らない場合は、通常の `Ctrl + Z` で操作全体を1ステップ戻してください。

Blender同梱のNumPyを使用します。大規模メッシュでは初回の形状解析に時間がかかります。形状が同じ間は解析結果を再利用し、Sculpt・Undo・接続変更・非表示変更後は再計算します。まず面の向きと接平面からの高さが連続する部分を元の領域へ確保し、盛り上がりが平面側へ侵食するのを抑えます。残る境界帯の所属は面に沿った実距離で決め、その帯の中だけで、広く平滑化した谷の強さと境界の実際の長さを使って線を整えます。画面の明るさは参照せず、視点や照明を変えても同じ境界を使います。メッシュの座標は変更しません。広い途切れや形状上区別できない境界は越える可能性があり、滑らかな領域が全くない部分はクリック面だけを対象にします。

大規模メッシュの初回準備では Edit Mode へ切り替えず、全 Face の center と全 loop の edge 次数だけを読み取ります。カーソル seed の Euclidean 範囲と解析用 halo を切り出し、範囲内の polygon の edge、法線、非表示状態だけで局所 CSR と形状判定を作ります。全体の曲率や partition を先に計算してから切り出す処理は行いません。edge の全体次数が2の共有だけを接続し、次数1の継ぎ目は両端点の一致、逆向き、法線の互換性を確認した場合だけ橋渡しします。局所切断境界と非多様体 edge は継ぎ目として扱いません。ホイール拡大時は保持済み配列・面 record を使って要求された局所 crop と record を再構築し、縮小と同じ半径への再訪では局所データを再利用します。

この機能は Sculpt Mode 専用です。`bpy.ops.sculpt.expand()`、Sculpt Mask、画面の深度や表裏で候補を決める処理、Face Set の新規 ID 生成は使用しません。準備中の処理は内部timerで区切られ、ready前のE再押下やEnterは確定しません。

ショートカットは Blender の `Preferences > Keymap` で `Mesh Focus: Local Face Set Grow` を検索して変更できます。

## Topology Colors

編集モードで面を選択し、上段の `Ctrl + Alt + 1`〜`6` を押すと、選択した表示中の面へ色番号を保存して半透明オーバーレイを表示します。`Ctrl + Alt + 0` は選択面の色を解除します。RetopoFlow 4 の PolyPen 待機中にもこの機能のキー割当が登録されます。選択を解除した後も色は残り、N パネルの `MFO > Topology Colors` から表示のON/OFF、透明度、割当、解除を操作できます。

色番号はマテリアルを作らず、active Edit Mesh の `mfo_topology_color` FACE 整数属性（0=解除、1〜6=色）へ保存します。.blend、Undo/Redoに含まれます。非表示面、未選択面、別オブジェクトの面、既存マテリアルは変更しません。面の境界と選択中の辺・頂点は読み分けられるように表示します。

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
- `Topology Colors`: 6色オーバーレイの表示と透明度

## 制限と復旧

- Reference Object が未指定の場合、通常 MFO と FSMFO は起動しません
- 画面中央に Reference Object の表面がない場合は起動しません
- FSMFO には Reference Object の `.sculpt_face_set` が必要です
- RetopoFlow と PolyQuilt の連携機能は、それぞれのアドオンがインストールされている場合だけ有効になります
- ファイルロード、アドオン無効化、ウィンドウ終了時には一時 Proxy、isolation、hook をクリーンアップします

---

# Mesh Focus Orbit — English

A Blender 5.2 add-on for manual retopology workflows.

Current add-on version: **3.2.24**

Main features:

- **Normal MFO**: in Object, Edit, or Sculpt Mode, temporarily orbits around the surface point at the center of the viewport
- **Face Set MFO (FSMFO)**: temporarily shows one Reference Face Set and its matching Retopo island
- **Smart Face Set Fill**: previews only the outer boundary under the cursor and applies the seed-connected faces inside it after a second E press or Enter
- Smart Face Set Fill draws boundary lines during prediction and resolves the generation snapshot into one seed flood at confirmation, preserving distance, hidden, crop, and mesh-domain limits
- **Topology Colors**: assigns six translucent topology guide colors to selected Edit Mesh faces

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
| Normal MFO ON/OFF | In Object, Edit, or Sculpt Mode, double-tap the configured key |
| Face Set MFO ON/OFF | In Object or Edit Mode, hold `Ctrl` and double-tap the configured key |
| Smart Face Set Fill preview | `E` in Sculpt Mode; release and press `E` again (or press `Enter`) to apply, wheel changes distance, `Esc` cancels |
| Strict Smart Face Set Fill preview | `Ctrl + E` in Sculpt Mode; release and press `E` again (or press `Enter`) to apply, wheel changes distance, `Esc` cancels |
| Topology Colors | `Ctrl + Alt + 1..6` in Edit Mode (`0` clears) |

Normal MFO and FSMFO use separate KeyMap Items. FSMFO is started directly by its Undo-enabled Activation Operator; it does not pass through an outer non-Undo trigger. The Activation Operator starts the non-Undo Watcher and then finishes.

## Normal MFO

1. Place the Reference Object at the center of the viewport (available in Object, Edit, and Sculpt Mode)
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

In Sculpt Mode, place the cursor over a Face Set and press `E` to start the local preview. Release the starting `E`, then press `E` again or press `Enter` to apply; `Esc` cancels. Holding the starting key or its repeat events cannot apply a partial result. Left and right clicks are consumed by the modal preview and do not apply or cancel it.

The algorithm:

- Uses the face under the cursor as the seed
- Reuses the seed's existing Face Set ID
- Traverses shared edges and matching unwelded seams without modifying topology
- Detects steps and valleys using smoothed normals and curvature at multiple scales
- Traverses convex crests, detects concave feet with directional curvature, and expands the boundary by one adjacency step
- Finds the smooth core, then assigns boundary faces to the nearest core without merging cores
- Skips hidden faces and non-manifold boundaries; a ray that first reaches a hidden face continues to the first visible face. There is no fixed 100,000-face cutoff; wheel-adjusted search distance is capped at 16 times the initial radius
- Writes the existing seed Face Set ID directly to the accepted faces
- Uses fixed view-independent multi-direction Lambert illumination as an approximation, not real shadows or ambient occlusion. It acquires the requested local distance patch first, detects signed valley changes there, keeps the seed-side original-face component across valley barriers, and corrects its surrounding band against original faces. Each radius is reevaluated from the current patch so a finite valley may reconnect naturally at its end.
- Labels one compact radius domain from its outside openings to remove inner boundary loops. Open U grooves, crop or mesh boundaries, non-manifold edges, hidden neighbors, unmatched seams, and separate sheets remain outside; a closed pocket connected to the candidate is included.
- Treats a valid shared-edge crossing whose outside face is within the current candidate distance as a shape/region boundary (orange); the fine partition barrier does not demote it to the distance boundary (cyan). Crossings outside the distance or beyond the explored patch remain distance boundaries.
- Uses the non-selected adjacent face for boundary distance and draws only copied boundary lines. No prediction triangulation or face GPU batch is created; confirmation floods the generation-fixed compact graph without crossing its stored outer boundary. History keeps the current and two previous stages.
- Wheel input received during synchronous computation and around the first draw of the latest result is discarded. After the draw callback and a short drain interval, the next wheel is accepted as one stage; Esc and E/Enter confirmation of a ready result remain available.

The first preparation stays in Sculpt Mode. It reads only all face centers and the global degree of every loop edge, then crops a cursor-centered Euclidean range with an analysis halo. Polygon edges, normals, and hidden flags are read only inside that range to build the local CSR, surface distances, valley bands, and boundaries. It does not build full-mesh curvature or partition data and crop it afterward. A normal shared edge is connected only when its global degree is two. A degree-one seam is bridged only after coincident endpoints, reverse winding, and compatible normals are confirmed. Crop boundaries and non-manifold edges are never treated as seams. Wheel expansion reuses retained arrays and face records while rebuilding the requested local crop and records; shrink and revisiting a prepared radius reuse the local data.

`Ctrl + E` enables Strict Mode with the same search extent and stricter boundary decisions. Use normal Blender `Ctrl + Z` to undo the complete operation in one step.

Uses Blender's bundled NumPy. Initial local preparation of large meshes can take several seconds; geometry results are reused until coordinates, topology, transforms or visibility change. Tangent-continuous core extensions are reserved first, using both source-anchored normal and plane-height tolerances; these extensions cannot be reassigned by contour smoothing. This keeps the flat base beyond a raised foot on the base side. Remaining boundary ownership uses physical surface distance, followed by up to six local contour-relaxation sweeps restricted to the boundary band. The energy combines physical boundary length with broadly smoothed concave normal changes; it does not sample screen brightness and is independent of view and lighting. Mesh coordinates are unchanged. Wide gaps or geometrically indistinguishable boundaries can still leak. Components with no smooth core fall back to the clicked face.

This feature is Sculpt Mode only. It does not use `bpy.ops.sculpt.expand()`, Sculpt Mask, screen depth/backface state to choose candidates, or newly generated Face Set IDs. Preparation is divided across internal timer ticks; E/Enter before Ready cannot apply a partial result.

The shortcut can be changed in Blender's `Preferences > Keymap` by searching for `Mesh Focus: Local Face Set Grow`.

## Topology Colors

In Edit Mode, select faces and press the top-row `Ctrl + Alt + 1` through `6` to store a color number on the selected visible faces and draw a translucent overlay. `Ctrl + Alt + 0` clears the selected faces. The feature's keymap entries are also registered while RetopoFlow 4 PolyPen is waiting. Colors remain visible after deselection. Use `MFO > Topology Colors` in the N-panel to toggle display, adjust opacity, assign colors, or clear them.

The color number is stored without creating materials, in the active Edit Mesh's `mfo_topology_color` FACE integer attribute (`0` cleared, `1` through `6` colored). It is included in `.blend` files and Blender Undo/Redo. Hidden faces, unselected faces, other objects, and existing materials are left unchanged. Face boundaries and selected edges and vertices remain distinguishable.

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
- `Topology Colors`: Toggle the six-color overlay and adjust its opacity

## Limitations and recovery

- Normal MFO and FSMFO do not start without a configured Reference Object
- They do not start when the viewport-center ray misses the Reference Object
- FSMFO requires the Reference Object's `.sculpt_face_set` attribute
- RetopoFlow and PolyQuilt integration is enabled only when the corresponding add-ons are installed
- Temporary Proxies, isolation state, and hooks are cleaned up during file loading, add-on disable, and window teardown
